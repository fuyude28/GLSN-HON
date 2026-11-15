"""Independent, fixed-budget physical-port bypass benefit experiments."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig

HERE = Path(__file__).resolve().parent
GLSN = HERE.parent
DATA = GLSN.parent / "data"
sys.path.insert(0, str(GLSN / "robustness"))

from robustness_experiment import (  # noqa: E402
    HistoricalStates, attack_schedule, neighbors, read_graph, read_order,
)

BUDGETS = (1, 3, 5)
MAIN_BUDGET = 3
ROUNDS = 16
SEED = 42
RESULTS = HERE / "results"


def area(values: list[float], attacked: list[int], n: int) -> float:
    return sum((a + b) * (q1 - q0) for a, b, q0, q1 in
               zip(values[:-1], values[1:], attacked[:-1], attacked[1:])) / (2 * n)


def port_curve(topology: ig.Graph, active_sets: list[list[int]],
               port_states: list[int], n_states: int) -> list[float]:
    """Fraction of this physical port's states in a multinational GSCC."""
    values = []
    for active in active_sets:
        components = topology.induced_subgraph(active).connected_components(mode="strong")
        sizes = components.sizes()
        largest = max(range(len(sizes)), key=sizes.__getitem__)
        membership = components.membership
        count = 0
        for state in port_states:
            index = bisect.bisect_left(active, state)
            if index < len(active) and active[index] == state and membership[index] == largest:
                count += 1
        # A component containing only this port is not globally integrated.
        values.append(count / n_states if sizes[largest] > count else 0.0)
    return values


def baseline_global_curves(topology: ig.Graph, active_sets: list[list[int]],
                            states: HistoricalStates,
                            port_sizes: dict[str, int]) -> dict[str, list[float]]:
    curves = {port: [] for port in port_sizes}
    for active in active_sets:
        components = topology.induced_subgraph(active).connected_components(mode="strong")
        sizes = components.sizes()
        largest = max(range(len(sizes)), key=sizes.__getitem__)
        members = components[largest]
        counts = Counter(states.ports[active[local]] for local in members)
        multinational = len(counts) > 1
        for port, values in curves.items():
            values.append(counts[port] / port_sizes[port] if multinational else 0.0)
    return curves


def add_port_bypasses(port: str, method: str, budget: int, rng: random.Random,
                      ranked_by_port: dict[str, list[tuple[int, int]]],
                      states: HistoricalStates, existing: set[tuple[int, int]],
                      incoming: list[set[int]], outgoing: list[set[int]],
                      in_degree: list[int], out_degree: list[int]) -> tuple[list[tuple[int, int]], list[dict]]:
    """One edge per ranked state with pi(v)=port; update topology and degrees."""
    extra_in: dict[int, set[int]] = defaultdict(set)
    extra_out: dict[int, set[int]] = defaultdict(set)
    added: list[tuple[int, int]] = []
    added_set: set[tuple[int, int]] = set()
    trace = []
    for rank, critical in ranked_by_port[port]:
        if len(added) >= budget:
            break
        sources = incoming[critical] | extra_in[critical]
        next_ports = {states.ports[w] for w in outgoing[critical] | extra_out[critical]
                      if w != critical}
        candidates = set()
        for source in sources:
            if source == critical:
                continue
            for next_port in next_ports:
                target = states.update(source, next_port)
                edge = source, target
                if (target != source and target != critical and edge not in existing
                        and edge not in added_set):
                    candidates.add(edge)
        if not candidates:
            continue
        ordered = sorted(candidates)
        if method == "CB-LDF":
            source, target = min(
                ordered, key=lambda e: (out_degree[e[0]] + len(extra_out[e[0]]) +
                                        in_degree[e[1]] + len(extra_in[e[1]]), e))
        elif method == "Bypass Random":
            source, target = rng.choice(ordered)
        else:
            raise ValueError(method)
        score = (out_degree[source] + len(extra_out[source]) +
                 in_degree[target] + len(extra_in[target]))
        edge = source, target
        added.append(edge)
        added_set.add(edge)
        extra_out[source].add(target)
        extra_in[target].add(source)
        trace.append(dict(port=port, method=method, edge_number=len(added),
                          priority_rank=rank, critical_state=states.labels[critical],
                          source_state=states.labels[source],
                          target_state=states.labels[target], degree_at_choice=score))
    return added, trace


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[dict], metadata: dict) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in results:
        groups[row["port"], row["method"], row["budget"], row["scenario"]].append(row)
    output = []
    for (port, method, budget, scenario), group in sorted(groups.items()):
        delta = [row["delta_r"] for row in group]
        absolute = [row["delta_a"] for row in group]
        mean = sum(delta) / len(delta)
        sd = (math.sqrt(sum((x - mean) ** 2 for x in delta) / (len(delta) - 1))
              if len(delta) > 1 else "")
        output.append(dict(port=port, method=method, budget=budget, scenario=scenario,
                           repeats=len(group), fo_total_degree=group[0]["fo_total_degree"],
                           state_count=group[0]["state_count"],
                           r_before=group[0]["r_before"],
                           mean_r_after=sum(row["r_after"] for row in group) / len(group),
                           mean_delta_r=mean, sd_delta_r=sd,
                           mean_delta_a=sum(absolute) / len(absolute)))
    write_rows(RESULTS / "port_summary.csv", output, list(output[0]))
    main_ports = {row["port"] for row in output if row["method"] == "CB-LDF"
                  and row["budget"] == MAIN_BUDGET and row["scenario"] == "global"}
    random_ports = {row["port"] for row in output if row["method"] == "Bypass Random"
                    and row["budget"] == MAIN_BUDGET and row["scenario"] == "global"
                    and row["repeats"] == metadata["random_repeats"]}
    metadata["actual_cb_ldf_ports_by_budget"] = {
        str(b): len({row["port"] for row in output if row["method"] == "CB-LDF"
                     and row["budget"] == b and row["scenario"] == "global"})
        for b in BUDGETS}
    metadata["main_ports"] = len(main_ports)
    metadata["bypass_random_complete_ports"] = len(random_ports)
    (RESULTS / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in
                      ("actual_cb_ldf_ports_by_budget", "main_ports",
                       "bypass_random_complete_ports")}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--ports", default="",
                        help="Optional comma-separated physical port IDs for a targeted run")
    args = parser.parse_args()
    if args.random_repeats < 1:
        parser.error("random-repeats must be positive")
    start = time.perf_counter()
    labels, edges = read_graph(DATA / "network-ho.csv")
    order = read_order(GLSN / "critical/DND-DRL/results/formal_removal_sequence.csv", labels)
    states = HistoricalStates(labels)
    n = len(labels)
    incoming, outgoing = neighbors(n, set(edges))
    existing = set(edges)
    in_degree = [len(x) for x in incoming]
    out_degree = [len(x) for x in outgoing]
    ranked_by_port: dict[str, list[tuple[int, int]]] = defaultdict(list)
    port_states: dict[str, list[int]] = defaultdict(list)
    for rank, node in enumerate(order, start=1):
        ranked_by_port[states.ports[node]].append((rank, node))
    for node, port in enumerate(states.ports):
        port_states[port].append(node)
    fo_labels, fo_edges = read_graph(DATA / "network-fo.csv")
    fo_graph = ig.Graph(n=len(fo_labels), edges=fo_edges, directed=True)
    first_order_degree = {label.split("|", 1)[0]: degree
                          for label, degree in zip(fo_labels, fo_graph.degree(mode="all"))}
    with (HERE / "port_eligibility.csv").open(encoding="utf-8", newline="") as handle:
        eligibility_rows = list(csv.DictReader(handle))
        eligible = {row["port"] for row in eligibility_rows
                    if int(row["eligible_1"])}
    eligible_main = {row["port"] for row in eligibility_rows
                     if int(row["eligible_3"])}
    if args.ports:
        eligible &= set(args.ports.split(","))
    ports = sorted(eligible, key=int)
    attacked, global_active = attack_schedule(n, order, ROUNDS)
    attacked_ports = {states.ports[node] for node in order[:attacked[-1]]}
    base = ig.Graph(n=n, edges=edges, directed=True)
    global_baseline = baseline_global_curves(base, global_active, states,
                                              {p: len(s) for p, s in port_states.items()})
    RESULTS.mkdir(exist_ok=True)
    records: list[dict] = []
    edge_rows: list[dict] = []
    curve_fields = ["port", "method", "repeat", "scenario", "budget", "attacked",
                    "fraction_in_gscc"]
    with (RESULTS / "port_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        curve_writer = csv.DictWriter(handle, fieldnames=curve_fields)
        curve_writer.writeheader()
        for count, port in enumerate(ports, start=1):
            own_states = port_states[port]
            if port in attacked_ports:
                foreign_order = [node for node in order if states.ports[node] != port]
                _, external_active = attack_schedule(n, foreign_order, ROUNDS)
            else:
                external_active = global_active
            before_curves = {
                "global": global_baseline[port],
                "external": (port_curve(base, external_active, own_states, len(own_states))
                             if external_active is not global_active else global_baseline[port]),
            }
            before = {mode: area(values, attacked, n)
                      for mode, values in before_curves.items()}
            for mode, values in before_curves.items():
                for q, value in zip(attacked, values):
                    curve_writer.writerow(dict(port=port, method="Baseline", repeat=0,
                                               scenario=mode, budget=0, attacked=q,
                                               fraction_in_gscc=value))
            evaluated: dict[tuple[tuple[int, int], ...], dict[str, list[float]]] = {}
            methods = [("CB-LDF", 0, 5)]
            # Compare the stochastic control on the exact main-budget sample.
            if port in eligible_main:
                methods.extend(("Bypass Random", repeat, MAIN_BUDGET)
                               for repeat in range(args.random_repeats))
            for method, repeat, max_budget in methods:
                rng = random.Random(SEED + repeat)
                added, trace = add_port_bypasses(
                    port, method, max_budget, rng, ranked_by_port, states, existing,
                    incoming, outgoing, in_degree, out_degree)
                edge_rows.extend(dict(repeat=repeat, **row) for row in trace)
                for budget in (BUDGETS if method == "CB-LDF" else (MAIN_BUDGET,)):
                    if len(added) < budget:
                        continue
                    key = tuple(sorted(added[:budget]))
                    if key not in evaluated:
                        topology = base.copy()
                        topology.add_edges(key)
                        global_values = port_curve(topology, global_active, own_states,
                                                   len(own_states))
                        external_values = (port_curve(topology, external_active, own_states,
                                                      len(own_states))
                                           if external_active is not global_active
                                           else global_values)
                        evaluated[key] = {"global": global_values,
                                          "external": external_values}
                    for mode in ("global", "external"):
                        values = evaluated[key][mode]
                        after = area(values, attacked, n)
                        if before[mode] <= 0:
                            continue
                        records.append(dict(port=port, method=method, repeat=repeat,
                                            scenario=mode, budget=budget,
                                            fo_total_degree=first_order_degree[port],
                                            state_count=len(own_states),
                                            r_before=before[mode], r_after=after,
                                            delta_a=after - before[mode],
                                            delta_r=(after - before[mode]) / before[mode]))
                        for q, value in zip(attacked, values):
                            curve_writer.writerow(dict(port=port, method=method,
                                                       repeat=repeat, scenario=mode,
                                                       budget=budget, attacked=q,
                                                       fraction_in_gscc=value))
            if count % 25 == 0 or count == len(ports):
                print(f"ports {count}/{len(ports)} elapsed={time.perf_counter() - start:.1f}s",
                      flush=True)

    write_rows(RESULTS / "port_trials.csv", records, list(records[0]))
    write_rows(RESULTS / "selected_edges.csv", edge_rows, list(edge_rows[0]))
    metadata = dict(graph="../network-ho.csv", first_order_degree_source="data/network-fo.csv",
                    order="../critical/DND-DRL/results/formal_removal_sequence.csv",
                    nodes=n, original_edges=len(edges),
                    initial_gscc=max(base.connected_components(mode="strong").sizes()),
                    total_physical_ports=len(port_states), eligible_ports_preflight=len(ports),
                    budgets=list(BUDGETS), main_budget=MAIN_BUDGET,
                    random_repeats=args.random_repeats, random_seeds=list(
                        range(SEED, SEED + args.random_repeats)),
                    attack_batch=attacked[1], attack_horizon=attacked[-1],
                    selection="one edge per ranked state with pi(v)=port; CB-LDF or uniform Bypass Random",
                    local_metric="fraction of all states of this physical port in a multinational GSCC",
                    area="trapezoidal partial area of local fraction versus attacked/initial N",
                    external_attack="skip all states of the focal port in saved DND order; take next foreign states")
    summarize(records, metadata)


if __name__ == "__main__":
    main()
