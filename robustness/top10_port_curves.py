"""Select and plot ten eligible ports with the largest mean CB-LDF gain."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import igraph as ig

from port_example_curves import (BUDGET, DATA, GLSN, HERE, RESULTS,
                                 add_tie_random_cb_ldf, port_names, write_csv)
from robustness_experiment import (HistoricalStates, attack_schedule, neighbors,
                                   read_graph, read_order)
import sys
sys.path.insert(0, str(GLSN / "rates"))
from run_rates import area, port_curve  # noqa: E402


PREFIX = "fig_a2_top10"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")

    with (GLSN / "rates/port_eligibility.csv").open(encoding="utf-8", newline="") as handle:
        eligibility = {row["port"]: row for row in csv.DictReader(handle)}
    candidates = sorted(
        (port for port, row in eligibility.items()
         if int(row["initial_gscc_states"]) > 0
         and int(row["candidate_rank_count"]) >= BUDGET), key=int)

    labels, edges = read_graph(DATA / "network-ho.csv")
    order = read_order(GLSN / "critical/DND-DRL/results/formal_removal_sequence.csv", labels)
    states = HistoricalStates(labels)
    base_edges = set(edges)
    base_in, base_out = neighbors(len(labels), base_edges)
    ranked: dict[str, list[tuple[int, int]]] = defaultdict(list)
    port_states: dict[str, list[int]] = defaultdict(list)
    for position, critical in enumerate(order, start=1):
        ranked[states.ports[critical]].append((position, critical))
    for node, port in enumerate(states.ports):
        port_states[port].append(node)
    names = port_names()
    base = ig.Graph(n=len(labels), edges=edges, directed=True)
    global_attacked, global_active = attack_schedule(len(labels), order)

    curves = []
    selected_edges = []
    summaries = []
    evaluated_ports = []
    for candidate_index, port in enumerate(candidates, start=1):
        own_states = port_states[port]
        foreign_order = [node for node in order if states.ports[node] != port]
        external_attacked, external_active = attack_schedule(len(labels), foreign_order)
        schedules = {
            "global": (global_attacked, global_active),
            "external": (external_attacked, external_active),
        }
        baseline = {mode: port_curve(base, active_sets, own_states, len(own_states))
                    for mode, (_, active_sets) in schedules.items()}
        runs = {mode: [] for mode in schedules}
        edge_sets = set()
        tied_choices = 0
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            added, trace = add_tie_random_cb_ldf(
                port, ranked, states, base_edges, base_in, base_out,
                random.Random(seed))
            if len(added) != BUDGET:
                raise ValueError(f"{names[port]}, seed {seed}: "
                                 f"only {len(added)} of {BUDGET} links added")
            edge_sets.add(frozenset(added))
            tied_choices += sum(row["minimum_score_ties"] > 1 for row in trace)
            for edge_number, row in enumerate(trace, start=1):
                selected_edges.append(dict(candidate_index=candidate_index, port=port,
                                           port_name=names[port], repeat=repeat,
                                           seed=seed, edge_number=edge_number, **row))
            augmented = base.copy()
            augmented.add_edges(added)
            for mode, (attack_counts, active) in schedules.items():
                after = port_curve(augmented, active, own_states, len(own_states))
                runs[mode].append(after)
                for q, before_value, after_value in zip(attack_counts, baseline[mode], after):
                    curves.append(dict(candidate_index=candidate_index, port=port,
                                       port_name=names[port],
                                       fo_total_degree=eligibility[port]["fo_total_degree"],
                                       state_count=len(own_states), scenario=mode,
                                       repeat=repeat, seed=seed, attacked=q,
                                       attack_fraction=q / len(labels),
                                       before=before_value, after=after_value))
        for mode, after_runs in runs.items():
            attack_counts = schedules[mode][0]
            before_area = area(baseline[mode], attack_counts, len(labels))
            after_areas = [area(curve, attack_counts, len(labels)) for curve in after_runs]
            gains = [(value - before_area) / before_area for value in after_areas]
            summaries.append(dict(candidate_index=candidate_index, port=port,
                                  port_name=names[port],
                                  fo_total_degree=eligibility[port]["fo_total_degree"],
                                  state_count=len(own_states), scenario=mode,
                                  repeats=args.repeats, before_auc=before_area,
                                  mean_after_auc=statistics.mean(after_areas),
                                  mean_delta_r=statistics.mean(gains),
                                  sd_delta_r=(statistics.stdev(gains)
                                              if len(gains) > 1 else ""),
                                  distinct_augmented_graphs=len(edge_sets)))
        evaluated_ports.append(dict(port=port, port_name=names[port],
                                     fo_total_degree=int(eligibility[port]["fo_total_degree"]),
                                     state_count=len(own_states),
                                     candidate_rank_count=int(eligibility[port]["candidate_rank_count"]),
                                     distinct_augmented_graphs=len(edge_sets),
                                     tied_choices=tied_choices,
                                     total_choices=args.repeats * BUDGET))
        print(f"{candidate_index:03d}/{len(candidates)} {names[port]}: "
              f"{len(edge_sets)} distinct graphs / "
              f"{args.repeats} trials", flush=True)

    scenario_gains = defaultdict(dict)
    for row in summaries:
        scenario_gains[row["port"]][row["scenario"]] = float(row["mean_delta_r"])
    ranked = sorted(candidates, key=lambda port: (
        -statistics.mean(scenario_gains[port][scenario]
                         for scenario in ("global", "external")), int(port)))
    gain_rank = {port: index for index, port in enumerate(ranked, start=1)}
    selected = ranked[:10]
    selected_ports = []
    for port in selected:
        record = next(row for row in evaluated_ports if row["port"] == port)
        selected_ports.append(dict(
            gain_rank=gain_rank[port], **record,
            global_mean_delta_r=scenario_gains[port]["global"],
            external_mean_delta_r=scenario_gains[port]["external"],
            combined_mean_delta_r=statistics.mean(
                scenario_gains[port][scenario] for scenario in ("global", "external"))))

    selected_curves = [dict(gain_rank=gain_rank[row["port"]], **row)
                       for row in curves if row["port"] in set(selected)]
    selected_edges_for_plot = [dict(gain_rank=gain_rank[row["port"]], **row)
                               for row in selected_edges if row["port"] in set(selected)]
    selected_summaries = [dict(gain_rank=gain_rank[row["port"]], **row)
                          for row in summaries if row["port"] in set(selected)]
    candidate_gains = []
    for port in ranked:
        candidate_gains.append(dict(
            gain_rank=gain_rank[port], port=port, port_name=names[port],
            fo_total_degree=eligibility[port]["fo_total_degree"],
            state_count=len(port_states[port]),
            global_mean_delta_r=scenario_gains[port]["global"],
            external_mean_delta_r=scenario_gains[port]["external"],
            combined_mean_delta_r=statistics.mean(
                scenario_gains[port][scenario] for scenario in ("global", "external"))))

    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "fig_a2_all_candidate_curves.csv", curves)
    write_csv(RESULTS / "fig_a2_all_candidate_selected_edges.csv", selected_edges)
    write_csv(RESULTS / "fig_a2_all_candidate_summary.csv", summaries)
    write_csv(RESULTS / "fig_a2_all_candidate_gains.csv", candidate_gains)
    write_csv(RESULTS / f"{PREFIX}_curves.csv", selected_curves)
    write_csv(RESULTS / f"{PREFIX}_selected_edges.csv", selected_edges_for_plot)
    write_csv(RESULTS / f"{PREFIX}_summary.csv", selected_summaries)
    protocol = dict(
        ranking="Top ten ports by equal-weight mean relative AUC gain across global and external attacks",
        ranking_population=len(candidates),
        eligibility="initial-GSCC states > 0 and at least ten eligible CB-LDF critical-state rankings",
        ranking_formula="combined_mean_delta_r = (global_mean_delta_r + external_mean_delta_r) / 2; each scenario mean is across the declared tie-randomization repeats",
        selection_uses_displayed_outcomes=True,
        source_graph="../network-ho.csv", nodes=len(labels),
        original_edges=len(edges), attack_horizon=global_attacked[-1],
        attack_horizon_by_scenario={
            "global": global_attacked[-1],
            "external_by_port": {port: len([node for node in order
                                             if states.ports[node] != port])
                                 for port in selected}},
        attack_sequence_length=len(order), attack_batch=math.ceil(len(labels) * 0.01),
        critical_order="../critical/DND-DRL/results/formal_removal_sequence.csv",
        selected_ports=selected_ports, budget_per_port=BUDGET,
        candidate_gain_summary="fig_a2_all_candidate_gains.csv",
        repeats=args.repeats,
        seeds=list(range(args.seed, args.seed + args.repeats)),
        method="CB-LDF: smallest current out-degree(source)+in-degree(target), random tie-break only",
        baseline="original graph, identical for all ten trials",
        attack="full saved DND-DRL dismantling sequence; batches of ceil(1% initial N), including a final partial batch; external skips all focal-port states",
        metric="fraction of focal physical port's states in multinational GSCC")
    (RESULTS / f"{PREFIX}_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
