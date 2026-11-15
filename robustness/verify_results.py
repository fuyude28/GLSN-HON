"""Check that saved bypass edges, attack curves and Fig. 8 data agree."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from robustness_experiment import HistoricalStates, neighbors, read_graph, read_order


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def rows(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    protocol = json.loads((RESULTS / "protocol.json").read_text(encoding="utf-8"))
    labels, base_edges = read_graph(HERE.parent.parent / "data/network-ho.csv")
    order = read_order(HERE.parent / "critical/DND-DRL/results/formal_removal_sequence.csv",
                       labels)
    index = {label: i for i, label in enumerate(labels)}
    states = HistoricalStates(labels)
    edge_groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows("selected_edges.csv"):
        edge_groups[row["method"], int(row["repeat"])].append(row)

    for (method, repeat), group in edge_groups.items():
        existing = set(base_edges)
        incoming, outgoing = neighbors(len(labels), existing)
        last_rank = 0
        for number, row in enumerate(group, start=1):
            u, z = index[row["source_state"]], index[row["target_state"]]
            assert int(row["edge_number"]) == number
            assert u != z and (u, z) not in existing, (method, repeat, number)
            if method != "Global Random":
                rank = int(row["priority_rank"])
                assert rank > last_rank and labels[order[rank - 1]] == row["critical_state"]
                v = order[rank - 1]
                assert u in incoming[v] and z != v
                assert any(states.update(u, states.ports[w]) == z
                           for w in outgoing[v] if w != v), (method, repeat, number)
                assert int(row["directed_degree_at_choice"]) == (
                    len(outgoing[u]) + len(incoming[z]))
                last_rank = rank
            existing.add((u, z))
            outgoing[u].add(z)
            incoming[z].add(u)
        assert len(group) == protocol["budgets"][-1], (method, repeat, len(group))

    curve_groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows("gscc_curves.csv"):
        curve_groups[row["method"], int(row["repeat"]), int(row["budget"])].append(row)
    trial_groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows("trials.csv"):
        key = row["method"], int(row["repeat"]), int(row["budget"])
        curve = curve_groups[key]
        q = [int(point["attacked"]) for point in curve]
        g = [int(point["gscc"]) for point in curve]
        n = len(labels)
        auc = sum((a + b) * (q1 - q0) for a, b, q0, q1 in
                  zip(g[:-1], g[1:], q[:-1], q[1:])) / (2 * n * n)
        assert len(curve) == protocol["attack_rounds"] + 1
        assert math.isclose(auc, float(row["auc"]), abs_tol=1e-12)
        assert math.isclose((auc - float(row["baseline_auc"])) /
                            float(row["baseline_auc"]), float(row["delta_r"]), abs_tol=1e-12)
        assert int(row["edges_added"]) == int(row["budget"])
        trial_groups[row["method"], int(row["budget"])].append(row)

    for row in rows("summary.csv"):
        group = trial_groups[row["method"], int(row["budget"])]
        values = [float(trial["delta_r"]) for trial in group]
        assert len(group) == int(row["trials"])
        assert math.isclose(sum(values) / len(values), float(row["mean_delta_r"]),
                            abs_tol=1e-12)
    assert len(edge_groups) == 2 + 4 * protocol["repeats"]
    assert len(curve_groups) == (2 + 4 * protocol["repeats"]) * len(protocol["budgets"])
    print(f"OK: {len(edge_groups)} runs, {sum(map(len, edge_groups.values()))} added edges, "
          f"{len(curve_groups)} attack curves")


if __name__ == "__main__":
    main()
