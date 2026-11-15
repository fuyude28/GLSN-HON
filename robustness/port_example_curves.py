"""Fig. A2-style before/after port robustness curves for CB-LDF bypasses."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import igraph as ig

HERE = Path(__file__).resolve().parent
GLSN = HERE.parent
DATA = GLSN.parent / "data"
sys.path.insert(0, str(GLSN / "rates"))

from run_rates import area, port_curve  # noqa: E402
from robustness_experiment import (  # noqa: E402
    HistoricalStates, attack_schedule, bypass_candidates, neighbors,
    read_graph, read_order,
)

ROUNDS = 16
BUDGET = 10
RESULTS = HERE / "results"


def select_examples() -> list[dict]:
    """Select three ports at fixed degree quantiles before observing gains."""
    with (GLSN / "rates/port_eligibility.csv").open(
            encoding="utf-8", newline="") as handle:
        eligible = [row for row in csv.DictReader(handle)
                    if int(row["initial_gscc_states"]) > 0
                    and int(row["candidate_rank_count"]) >= BUDGET]
    eligible.sort(key=lambda row: (int(row["fo_total_degree"]), int(row["port"])))
    positions = [round((len(eligible) - 1) * fraction)
                 for fraction in (0, 0.25, 0.75)]
    return [eligible[position] for position in positions]


def port_names() -> dict[str, str]:
    with (DATA / "port_system_mapping.csv").open(encoding="utf-8-sig", newline="") as handle:
        return {row["system_id"]: row["system_name_en"]
                for row in csv.DictReader(handle)}


def add_tie_random_cb_ldf(
    port: str, ranked: dict[str, list[tuple[int, int]]],
    states: HistoricalStates, base_edges: set[tuple[int, int]],
    base_in: list[set[int]], base_out: list[set[int]], rng: random.Random,
) -> tuple[list[tuple[int, int]], list[dict]]:
    existing = set(base_edges)
    incoming = [set(neighbors) for neighbors in base_in]
    outgoing = [set(neighbors) for neighbors in base_out]
    added = []
    trace = []
    for rank, critical in ranked[port]:
        if len(added) >= BUDGET:
            break
        candidates = bypass_candidates(critical, states, incoming, outgoing, existing)
        if not candidates:
            continue
        score = lambda edge: len(outgoing[edge[0]]) + len(incoming[edge[1]])
        minimum = min(map(score, candidates))
        tied = [edge for edge in candidates if score(edge) == minimum]
        source, target = rng.choice(tied)
        existing.add((source, target))
        outgoing[source].add(target)
        incoming[target].add(source)
        added.append((source, target))
        trace.append(dict(priority_rank=rank, critical_state=states.labels[critical],
                          source_state=states.labels[source],
                          target_state=states.labels[target], degree_score=minimum,
                          minimum_score_ties=len(tied)))
    return added, trace


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")

    labels, edges = read_graph(DATA / "network-ho.csv")
    order = read_order(GLSN / "critical/DND-DRL/results/formal_removal_sequence.csv", labels)
    states = HistoricalStates(labels)
    base_edges = set(edges)
    base_in, base_out = neighbors(len(labels), base_edges)
    ranked: dict[str, list[tuple[int, int]]] = defaultdict(list)
    port_states: dict[str, list[int]] = defaultdict(list)
    for rank, critical in enumerate(order, start=1):
        ranked[states.ports[critical]].append((rank, critical))
    for node, port in enumerate(states.ports):
        port_states[port].append(node)
    examples = select_examples()
    names = port_names()
    base = ig.Graph(n=len(labels), edges=edges, directed=True)
    attacked, global_active = attack_schedule(len(labels), order, ROUNDS)

    curves = []
    links = []
    summaries = []
    selected = []
    for example in examples:
        port = example["port"]
        foreign_order = [node for node in order if states.ports[node] != port]
        _, external_active = attack_schedule(len(labels), foreign_order, ROUNDS)
        schedules = {"global": global_active, "external": external_active}
        baseline = {mode: port_curve(base, active, port_states[port],
                                     len(port_states[port]))
                    for mode, active in schedules.items()}
        runs = {mode: [] for mode in schedules}
        edge_sets = set()
        ties = []
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            added, trace = add_tie_random_cb_ldf(
                port, ranked, states, base_edges, base_in, base_out,
                random.Random(seed))
            if len(added) != BUDGET:
                raise ValueError(f"port {port}, seed {seed}: only {len(added)} of "
                                 f"{BUDGET} bypass links are feasible")
            edge_sets.add(frozenset(added))
            ties.extend(row["minimum_score_ties"] for row in trace)
            for edge_number, row in enumerate(trace, start=1):
                links.append(dict(port=port, port_name=names[port], repeat=repeat,
                                  seed=seed, edge_number=edge_number, **row))
            augmented = base.copy()
            augmented.add_edges(added)
            for mode, active in schedules.items():
                after = port_curve(augmented, active, port_states[port],
                                   len(port_states[port]))
                runs[mode].append(after)
                for q, before_value, after_value in zip(attacked, baseline[mode], after):
                    curves.append(dict(port=port, port_name=names[port],
                                       fo_total_degree=example["fo_total_degree"],
                                       state_count=len(port_states[port]),
                                       scenario=mode, repeat=repeat, seed=seed,
                                       attacked=q, attack_fraction=q / len(labels),
                                       before=before_value, after=after_value))
        for mode, after_runs in runs.items():
            before_area = area(baseline[mode], attacked, len(labels))
            after_areas = [area(curve, attacked, len(labels)) for curve in after_runs]
            gains = [(value - before_area) / before_area for value in after_areas]
            summaries.append(dict(port=port, port_name=names[port],
                                  fo_total_degree=example["fo_total_degree"],
                                  state_count=len(port_states[port]), scenario=mode,
                                  repeats=args.repeats, before_auc=before_area,
                                  mean_after_auc=statistics.mean(after_areas),
                                  mean_delta_r=statistics.mean(gains),
                                  sd_delta_r=(statistics.stdev(gains)
                                              if len(gains) > 1 else ""),
                                  distinct_augmented_graphs=len(edge_sets)))
        selected.append(dict(port=port, port_name=names[port],
                             fo_total_degree=int(example["fo_total_degree"]),
                             state_count=len(port_states[port]),
                             candidate_rank_count=int(example["candidate_rank_count"]),
                             distinct_augmented_graphs=len(edge_sets),
                             tied_choices=sum(value > 1 for value in ties),
                             total_choices=len(ties)))
        print(f"{names[port]} ({port}, degree={example['fo_total_degree']}): "
              f"{len(edge_sets)} distinct edge sets in {args.repeats} trials", flush=True)

    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "fig_a2_port_curves.csv", curves)
    write_csv(RESULTS / "fig_a2_selected_edges.csv", links)
    write_csv(RESULTS / "fig_a2_summary.csv", summaries)
    protocol = dict(source_graph="../network-ho.csv", nodes=len(labels),
                    original_edges=len(edges), attack_horizon=attacked[-1],
                    critical_order=
                    "../critical/DND-DRL/results/formal_removal_sequence.csv",
                    example_selection="0th, 25th and 75th degree percentiles among ten-link-eligible ports, selected before observing their gains",
                    selected_ports=selected, budget_per_port=BUDGET,
                    repeats=args.repeats, seeds=list(range(args.seed, args.seed + args.repeats)),
                    method="CB-LDF: smallest current out-degree(source)+in-degree(target), random tie-break only",
                    baseline="original graph, identical for all ten trials",
                    attack="saved DND-DRL order, 16 fixed batches of 134 states; external skips all focal-port states",
                    metric="fraction of focal physical port's states in multinational GSCC")
    (RESULTS / "fig_a2_protocol.json").write_text(json.dumps(protocol, indent=2),
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
