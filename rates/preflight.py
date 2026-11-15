"""Count physical ports eligible for fixed-budget, port-centered bypasses."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig

HERE = Path(__file__).resolve().parent
GLSN = HERE.parent
DATA = GLSN.parent / "data"
sys.path.insert(0, str(GLSN / "robustness"))

from robustness_experiment import (  # noqa: E402
    HistoricalStates, bypass_candidates, neighbors, read_graph, read_order,
)


def main() -> None:
    labels, edges = read_graph(DATA / "network-ho.csv")
    order = read_order(GLSN / "critical/DND-DRL/results/formal_removal_sequence.csv", labels)
    states = HistoricalStates(labels)
    incoming, outgoing = neighbors(len(labels), set(edges))
    components = ig.Graph(n=len(labels), edges=edges, directed=True).connected_components(
        mode="strong")
    initial_core = set(max(components, key=len))

    state_counts = Counter(states.ports)
    core_counts = Counter(states.ports[node] for node in initial_core)
    ranked_counts = Counter(states.ports[node] for node in order)
    eligible_ranks: dict[str, list[int]] = defaultdict(list)
    existing = set(edges)
    for rank, critical in enumerate(order, start=1):
        if bypass_candidates(critical, states, incoming, outgoing, existing):
            eligible_ranks[states.ports[critical]].append(rank)

    fo_labels, fo_edges = read_graph(DATA / "network-fo.csv")
    fo_graph = ig.Graph(n=len(fo_labels), edges=fo_edges, directed=True)
    degrees = {label.split("|", 1)[0]: degree
               for label, degree in zip(fo_labels, fo_graph.degree(mode="all"))}
    rows = []
    for port in sorted(state_counts, key=int):
        ranks = eligible_ranks[port]
        row = dict(port=port, fo_total_degree=degrees[port],
                   state_count=state_counts[port], initial_gscc_states=core_counts[port],
                   ranked_state_count=ranked_counts[port], candidate_rank_count=len(ranks))
        for budget in (1, 3, 5):
            row[f"eligible_{budget}"] = int(core_counts[port] > 0 and len(ranks) >= budget)
            row[f"rank_to_{budget}"] = ranks[budget - 1] if len(ranks) >= budget else ""
        rows.append(row)
    with (HERE / "port_eligibility.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = dict(states=len(labels), edges=len(edges), ports=len(rows),
                   ports_in_initial_gscc=sum(row["initial_gscc_states"] > 0 for row in rows),
                   eligible={str(b): sum(row[f"eligible_{b}"] for row in rows)
                             for b in (1, 3, 5)},
                   selection="one history-consistent edge per ranked state whose physical port is the target port",
                   eligibility_basis="candidate ranks in the unchanged original graph")
    (HERE / "eligibility_summary.json").write_text(json.dumps(summary, indent=2),
                                                    encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
