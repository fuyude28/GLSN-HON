"""Verify full-sequence gains and selection of the ten highest-gain ports."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict

from port_example_curves import BUDGET, DATA, GLSN, RESULTS, add_tie_random_cb_ldf
from robustness_experiment import (DATA, HistoricalStates, attack_schedule, neighbors,
                                   read_graph, read_order)
from top10_port_curves import PREFIX
from run_rates import area


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    protocol = json.loads((RESULTS / f"{PREFIX}_protocol.json").read_text(encoding="utf-8"))
    labels, edges = read_graph(DATA / "network-ho.csv")
    order = read_order(GLSN / "critical/DND-DRL/results/formal_removal_sequence.csv", labels)
    states = HistoricalStates(labels)
    base_edges = set(edges)
    base_in, base_out = neighbors(len(labels), base_edges)
    ranked_states = defaultdict(list)
    for rank, critical in enumerate(order, start=1):
        ranked_states[states.ports[critical]].append((rank, critical))

    eligible = read_csv(GLSN / "rates/port_eligibility.csv")
    candidates = sorted((row["port"] for row in eligible
                         if int(row["initial_gscc_states"]) > 0
                         and int(row["candidate_rank_count"]) >= BUDGET), key=int)
    assert len(candidates) == protocol["ranking_population"]
    assert protocol["attack_sequence_length"] == len(order)

    all_curves = defaultdict(list)
    for row in read_csv(RESULTS / "fig_a2_all_candidate_curves.csv"):
        all_curves[row["port"], row["scenario"], int(row["repeat"])].append(row)
    assert len(all_curves) == len(candidates) * 2 * protocol["repeats"]

    candidate_gains = read_csv(RESULTS / "fig_a2_all_candidate_gains.csv")
    assert len(candidate_gains) == len(candidates)
    calculated = {}
    candidate_summaries = {(row["port"], row["scenario"]): row
                           for row in read_csv(RESULTS / "fig_a2_all_candidate_summary.csv")}
    assert len(candidate_summaries) == len(candidates) * 2
    for port in candidates:
        foreign_order = [node for node in order if states.ports[node] != port]
        expected = {
            "global": attack_schedule(len(labels), order)[0],
            "external": attack_schedule(len(labels), foreign_order)[0],
        }
        scenario_gains = {}
        for scenario in ("global", "external"):
            before_areas, after_areas = [], []
            for repeat in range(protocol["repeats"]):
                points = all_curves[port, scenario, repeat]
                q = [int(row["attacked"]) for row in points]
                assert q == expected[scenario]
                before = [float(row["before"]) for row in points]
                after = [float(row["after"]) for row in points]
                before_areas.append(area(before, q, len(labels)))
                after_areas.append(area(after, q, len(labels)))
            assert max(before_areas) - min(before_areas) < 1e-15
            base_area = before_areas[0]
            gains = [(value - base_area) / base_area for value in after_areas]
            summary = candidate_summaries[port, scenario]
            assert math.isclose(base_area, float(summary["before_auc"]), abs_tol=1e-12)
            assert math.isclose(statistics.mean(after_areas),
                                float(summary["mean_after_auc"]), abs_tol=1e-12)
            assert math.isclose(statistics.mean(gains),
                                float(summary["mean_delta_r"]), abs_tol=1e-12)
            scenario_gains[scenario] = statistics.mean(gains)
        calculated[port] = (scenario_gains["global"] + scenario_gains["external"]) / 2

    ranked_ports = sorted(candidates, key=lambda port: (-calculated[port], int(port)))
    assert [row["port"] for row in candidate_gains] == ranked_ports
    for rank, row in enumerate(candidate_gains, start=1):
        assert int(row["gain_rank"]) == rank
        assert math.isclose(float(row["combined_mean_delta_r"]),
                            calculated[row["port"]], abs_tol=1e-12)
    selected = protocol["selected_ports"]
    assert [row["port"] for row in selected] == ranked_ports[:10]

    selected_curves = read_csv(RESULTS / f"{PREFIX}_curves.csv")
    expected_selected = [row for row in read_csv(RESULTS / "fig_a2_all_candidate_curves.csv")
                         if row["port"] in set(ranked_ports[:10])]
    assert len(selected_curves) == len(expected_selected)
    all_edges = read_csv(RESULTS / "fig_a2_all_candidate_selected_edges.csv")
    selected_edges = read_csv(RESULTS / f"{PREFIX}_selected_edges.csv")
    assert len(selected_edges) == len(protocol["selected_ports"]) * protocol["repeats"] * BUDGET
    assert {(r["port"], r["repeat"], r["edge_number"], r["source_state"], r["target_state"])
            for r in selected_edges} == {
                (r["port"], r["repeat"], r["edge_number"], r["source_state"], r["target_state"])
                for r in all_edges if r["port"] in set(ranked_ports[:10])}

    links_by_trial = defaultdict(list)
    for row in selected_edges:
        links_by_trial[row["port"], int(row["repeat"])].append(row)
    for (port, repeat), group in links_by_trial.items():
        assert len(group) == BUDGET
        added, trace = add_tie_random_cb_ldf(
            port, ranked_states,
            states=states, base_edges=base_edges, base_in=base_in, base_out=base_out,
            rng=random.Random(protocol["seeds"][repeat]))
        assert len(added) == BUDGET
        for row, (source, target), choice in zip(group, added, trace):
            assert row["source_state"] == labels[source]
            assert row["target_state"] == labels[target]
            assert row["critical_state"] == choice["critical_state"]
            assert int(row["degree_score"]) == choice["degree_score"]

    assert (RESULTS / f"{PREFIX}_port_curves.png").stat().st_size > 0
    print(f"OK: ranked {len(candidates)} eligible ports; selected 10; "
          f"{len(links_by_trial)} selected trials and "
          f"{len(selected_curves)} selected curve rows")


if __name__ == "__main__":
    main()
