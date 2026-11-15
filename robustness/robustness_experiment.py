"""Critical-state bypass link addition on the directed higher-order GLSN.

The saved DND-DRL order is used both to prioritize bypasses and as a fixed,
common attack sequence. This measures robustness under that attack, rather
than retraining or reranking the attacker for each augmented graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import igraph as ig


HERE = Path(__file__).resolve().parent
GLSN = HERE.parent
DATA = GLSN.parent / "data"
ORDER = GLSN / "critical/DND-DRL/results/formal_removal_sequence.csv"
METHODS = ("CB-LDF", "CB-LP", "CB-HDF", "CB-HP", "Bypass Random", "Global Random")
STOCHASTIC = {"CB-LP", "CB-HP", "Bypass Random", "Global Random"}


def read_graph(path: Path) -> tuple[list[str], list[tuple[int, int]]]:
    labels: list[str] = []
    index: dict[str, int] = {}
    edges: set[tuple[int, int]] = set()

    def vertex(label: str) -> int:
        if label not in index:
            index[label] = len(labels)
            labels.append(label)
        return index[label]

    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            edges.add((vertex(row["FromNode"]), vertex(row["ToNode"])))
    return labels, sorted(edges)


def read_order(path: Path, labels: list[str]) -> list[int]:
    index = {label: i for i, label in enumerate(labels)}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        order = [index[row["state_node"]] for row in csv.DictReader(handle)]
    if len(order) != len(set(order)):
        raise ValueError("DND-DRL removal order contains duplicate states")
    return order


class HistoricalStates:
    """Resolve the longest existing suffix after appending a physical port."""

    def __init__(self, labels: list[str]):
        self.labels = labels
        self.by_label = {label: i for i, label in enumerate(labels)}
        self.ports: list[str] = []
        self.reversed_histories: list[tuple[str, ...]] = []
        self.max_order = 1
        for label in labels:
            port, previous = label.split("|", 1)
            history = (port, *(previous.split(".") if previous else ()))
            self.ports.append(port)
            self.reversed_histories.append(history)
            self.max_order = max(self.max_order, len(history))
        self.cache: dict[tuple[int, str], int] = {}

    def update(self, source: int, destination_port: str) -> int:
        key = source, destination_port
        if key in self.cache:
            return self.cache[key]
        history = self.reversed_histories[source]
        for length in range(min(len(history), self.max_order - 1), -1, -1):
            suffix = ".".join(history[:length])
            label = destination_port + "|" + suffix
            if label in self.by_label:
                result = self.by_label[label]
                self.cache[key] = result
                return result
        raise ValueError(f"No first-order state for physical port {destination_port}")


def neighbors(n: int, edges: set[tuple[int, int]]) -> tuple[list[set[int]], list[set[int]]]:
    incoming = [set() for _ in range(n)]
    outgoing = [set() for _ in range(n)]
    for u, v in edges:
        outgoing[u].add(v)
        incoming[v].add(u)
    return incoming, outgoing


def bypass_candidates(
    critical: int, state: HistoricalStates,
    incoming: list[set[int]], outgoing: list[set[int]],
    existing: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    destination_ports = {state.ports[w] for w in outgoing[critical] if w != critical}
    candidates = set()
    for u in incoming[critical]:
        if u == critical:
            continue
        for port in destination_ports:
            z = state.update(u, port)
            edge = (u, z)
            if z != u and z != critical and edge not in existing:
                candidates.add(edge)
    return sorted(candidates)


def choose_candidate(
    candidates: list[tuple[int, int]], method: str,
    out_degree: list[int], in_degree: list[int], rng: random.Random,
) -> tuple[int, int]:
    def degree(edge: tuple[int, int]) -> int:
        return out_degree[edge[0]] + in_degree[edge[1]]

    if method == "CB-LDF":
        return min(candidates, key=lambda e: (degree(e), e))
    if method == "CB-HDF":
        return min(candidates, key=lambda e: (-degree(e), e))
    if method == "Bypass Random":
        return rng.choice(candidates)
    if method == "CB-LP":
        weights = [1.0 / (degree(e) + 1) for e in candidates]
    elif method == "CB-HP":
        weights = [degree(e) + 1 for e in candidates]
    else:
        raise ValueError(method)
    return rng.choices(candidates, weights=weights, k=1)[0]


def add_links(
    labels: list[str], base_edges: list[tuple[int, int]], order: list[int],
    method: str, budget: int, rng: random.Random,
) -> tuple[list[tuple[int, int]], list[dict]]:
    n = len(labels)
    existing = set(base_edges)
    added: list[tuple[int, int]] = []
    trace: list[dict] = []

    if method == "Global Random":
        while len(added) < budget:
            u, z = rng.randrange(n), rng.randrange(n)
            if u == z or (u, z) in existing:
                continue
            existing.add((u, z))
            added.append((u, z))
            trace.append(dict(priority_rank="", critical_state="", source_state=labels[u],
                              target_state=labels[z], directed_degree_at_choice=""))
        return added, trace

    state = HistoricalStates(labels)
    incoming, outgoing = neighbors(n, existing)
    out_degree = [len(adjacency) for adjacency in outgoing]
    in_degree = [len(adjacency) for adjacency in incoming]
    for rank, critical in enumerate(order, start=1):
        if len(added) >= budget:
            break
        candidates = bypass_candidates(critical, state, incoming, outgoing, existing)
        if not candidates:
            continue
        u, z = choose_candidate(candidates, method, out_degree, in_degree, rng)
        score = out_degree[u] + in_degree[z]
        edge = (u, z)
        added.append(edge)
        existing.add(edge)
        outgoing[u].add(z)
        incoming[z].add(u)
        out_degree[u] += 1
        in_degree[z] += 1
        trace.append(dict(priority_rank=rank, critical_state=labels[critical],
                          source_state=labels[u], target_state=labels[z],
                          directed_degree_at_choice=score))
    return added, trace


def attack_schedule(
    n: int, order: list[int], rounds: int | None = None,
) -> tuple[list[int], list[list[int]]]:
    batch = math.ceil(n * 0.01)
    if rounds is None:
        rounds = math.ceil(len(order) / batch)
    elif len(order) < rounds * batch:
        raise ValueError("saved DND-DRL sequence is too short for the attack horizon")
    active = set(range(n))
    schedule = [sorted(active)]
    attacked = [0]
    for step in range(rounds):
        batch_order = order[step * batch:min((step + 1) * batch, len(order))]
        active.difference_update(batch_order)
        schedule.append(sorted(active))
        attacked.append(attacked[-1] + len(batch_order))
    return attacked, schedule


def evaluate(
    n: int, edges: list[tuple[int, int]], attacked: list[int],
    active_sets: list[list[int]],
) -> tuple[float, list[int]]:
    topology = ig.Graph(n=n, edges=edges, directed=True)
    gscc = []
    for active in active_sets:
        residual = topology.induced_subgraph(active)
        gscc.append(max(residual.connected_components(mode="strong").sizes(), default=0))
    # Trapezoidal partial area of GSCC/N versus attacked/N on a fixed horizon.
    area = sum((a + b) * (q1 - q0) for a, b, q0, q1 in
               zip(gscc[:-1], gscc[1:], attacked[:-1], attacked[1:])) / (2 * n * n)
    return area, gscc


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DATA / "network-ho.csv")
    parser.add_argument("--order", type=Path, default=ORDER)
    parser.add_argument("--budgets", default="50,100,150,200,250,300,350,400")
    parser.add_argument("--repeats", type=int, default=100,
                        help="Stochastic trials per strategy; deterministic strategies run once")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack-rounds", type=int, default=16)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()

    budgets = sorted({int(value) for value in args.budgets.split(",")})
    if not budgets or budgets[0] < 1 or args.repeats < 1 or args.attack_rounds < 1:
        parser.error("budgets, repeats and attack-rounds must be positive")
    labels, base_edges = read_graph(args.graph)
    order = read_order(args.order, labels)
    n = len(labels)
    attacked, active_sets = attack_schedule(n, order, args.attack_rounds)
    baseline_area, baseline_gscc = evaluate(n, base_edges, attacked, active_sets)
    if baseline_area <= 0:
        raise ValueError("baseline partial robustness area must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    trials: list[dict] = []
    traces: list[dict] = []
    curves: list[dict] = []
    for method in METHODS:
        repetitions = args.repeats if method in STOCHASTIC else 1
        for repeat in range(repetitions):
            seed = args.seed + repeat
            rng = random.Random(seed)
            added, trace = add_links(labels, base_edges, order, method, budgets[-1], rng)
            for edge_number, row in enumerate(trace, start=1):
                traces.append(dict(method=method, repeat=repeat, seed=seed,
                                   edge_number=edge_number, **row))
            for budget in budgets:
                prefix = added[:budget]
                area, gscc = evaluate(n, base_edges + prefix, attacked, active_sets)
                trials.append(dict(method=method, repeat=repeat, seed=seed, budget=budget,
                                   edges_added=len(prefix), baseline_auc=baseline_area,
                                   auc=area, delta_r=(area - baseline_area) / baseline_area,
                                   final_gscc=gscc[-1]))
                for q, size in zip(attacked, gscc):
                    curves.append(dict(method=method, repeat=repeat, seed=seed,
                                       budget=budget, attacked=q, gscc=size))
            print(f"{method} run {repeat + 1}/{repetitions}: {len(added)} links", flush=True)

    summary: list[dict] = []
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in trials:
        grouped[row["method"], row["budget"]].append(row)
    for method in METHODS:
        for budget in budgets:
            group = grouped[method, budget]
            values = [row["delta_r"] for row in group]
            mean = sum(values) / len(values)
            sd = (math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
                  if len(values) > 1 else None)
            summary.append(dict(method=method, budget=budget, trials=len(group),
                                min_edges_added=min(row["edges_added"] for row in group),
                                mean_edges_added=sum(row["edges_added"] for row in group) / len(group),
                                mean_delta_r=mean, sd_delta_r=sd))

    write_csv(args.out / "trials.csv", trials, list(trials[0]))
    write_csv(args.out / "summary.csv", summary, list(summary[0]))
    write_csv(args.out / "selected_edges.csv", traces, list(traces[0]))
    write_csv(args.out / "gscc_curves.csv", curves, list(curves[0]))
    write_csv(args.out / "baseline_curve.csv",
              [dict(attacked=q, gscc=size) for q, size in zip(attacked, baseline_gscc)],
              ["attacked", "gscc"])
    metadata = dict(graph=str(args.graph), order=str(args.order), nodes=n,
                    original_edges=len(base_edges), original_gscc=baseline_gscc[0],
                    baseline_partial_auc=baseline_area, budgets=budgets, repeats=args.repeats,
                    seed=args.seed, attack_rounds=args.attack_rounds,
                    attack_batch=attacked[1], attack_horizon=attacked[-1],
                    attack_protocol="fixed saved DND-DRL order; 1% of original N per round",
                    area_definition="trapezoid of GSCC/N versus removed/N, Q=0 to fixed horizon",
                    priority_protocol="one bypass edge per ranked critical state; skip empty pools",
                    candidate_protocol="current directed 2-step paths; history-consistent target",
                    bypass_random_protocol="uniform choice within the current critical state's history-consistent candidate set; one edge per ranked state",
                    global_random_protocol="uniform non-edge ordered state pair, no history constraint")
    (args.out / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"baseline_auc": baseline_area, "summary": str(args.out / "summary.csv")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
