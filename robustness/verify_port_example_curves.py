"""Verify Fig. A2-style selected links and saved before/after areas."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLSN = HERE.parent
DATA = GLSN.parent / "data"
RESULTS = HERE / "results"
sys.path.insert(0, str(GLSN / "rates"))

from run_rates import area  # noqa: E402
from robustness_experiment import (  # noqa: E402
    HistoricalStates, attack_schedule, bypass_candidates, neighbors,
    read_graph, read_order,
)


def rows(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    protocol = json.loads((RESULTS / "fig_a2_protocol.json").read_text(encoding="utf-8"))
    labels, edges = read_graph(DATA / "network-ho.csv")
    order = read_order(GLSN / "critical/DND-DRL/results/formal_removal_sequence.csv", labels)
    state = HistoricalStates(labels)
    index = {label: i for i, label in enumerate(labels)}
    base = set(edges)
    base_in, base_out = neighbors(len(labels), base)
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows("fig_a2_selected_edges.csv"):
        groups[row["port"], int(row["repeat"])].append(row)
    distinct: dict[str, set[frozenset]] = defaultdict(set)
    for (port, repeat), group in groups.items():
        existing = set(base)
        incoming = [set(values) for values in base_in]
        outgoing = [set(values) for values in base_out]
        selected = set()
        last_rank = 0
        for number, row in enumerate(group, start=1):
            rank = int(row["priority_rank"])
            critical = order[rank - 1]
            edge = index[row["source_state"]], index[row["target_state"]]
            candidates = bypass_candidates(critical, state, incoming, outgoing, existing)
            assert row["critical_state"] == labels[critical]
            assert state.ports[critical] == port and rank > last_rank
            assert int(row["edge_number"]) == number and edge in candidates
            minimum = min(len(outgoing[u]) + len(incoming[z]) for u, z in candidates)
            tied = sum(len(outgoing[u]) + len(incoming[z]) == minimum
                       for u, z in candidates)
            assert len(outgoing[edge[0]]) + len(incoming[edge[1]]) == minimum
            assert int(row["degree_score"]) == minimum
            assert int(row["minimum_score_ties"]) == tied
            existing.add(edge)
            outgoing[edge[0]].add(edge[1])
            incoming[edge[1]].add(edge[0])
            selected.add(edge)
            last_rank = rank
        assert len(group) == len(selected) == protocol["budget_per_port"]
        distinct[port].add(frozenset(selected))

    curves: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows("fig_a2_port_curves.csv"):
        curves[row["port"], row["scenario"], int(row["repeat"])].append(row)
    for row in rows("fig_a2_summary.csv"):
        port, scenario = row["port"], row["scenario"]
        before_areas = []
        after_areas = []
        for repeat in range(protocol["repeats"]):
            points = curves[port, scenario, repeat]
            assert len(points) == 17
            q = [int(point["attacked"]) for point in points]
            assert q[0] == 0 and q[-1] == protocol["attack_horizon"]
            before = [float(point["before"]) for point in points]
            after = [float(point["after"]) for point in points]
            assert all(0 <= value <= 1 for value in before + after)
            before_areas.append(area(before, q, len(labels)))
            after_areas.append(area(after, q, len(labels)))
        assert len(set(before_areas)) == 1
        assert math.isclose(before_areas[0], float(row["before_auc"]), abs_tol=1e-12)
        assert math.isclose(statistics.mean(after_areas), float(row["mean_after_auc"]),
                            abs_tol=1e-12)
        gain = statistics.mean((a - before_areas[0]) / before_areas[0]
                               for a in after_areas)
        assert math.isclose(gain, float(row["mean_delta_r"]), abs_tol=1e-12)
        assert len(distinct[port]) == int(row["distinct_augmented_graphs"])
    assert len(groups) == len(protocol["selected_ports"]) * protocol["repeats"]
    assert (RESULTS / "fig_a2_port_curves.png").stat().st_size > 0
    print(f"OK: {len(distinct)} ports, {len(groups)} trials, "
          f"{sum(map(len, groups.values()))} links, {sum(map(len, curves.values()))} points")


if __name__ == "__main__":
    main()
