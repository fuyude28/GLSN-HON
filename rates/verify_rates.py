"""Check port attribution, history-consistent edges and reported areas."""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLSN = HERE.parent
DATA = GLSN.parent / "data"
sys.path.insert(0, str(GLSN / "robustness"))

from robustness_experiment import HistoricalStates, neighbors, read_graph, read_order  # noqa: E402
from run_rates import area  # noqa: E402


def csv_rows(name: str) -> list[dict]:
    with (HERE / "results" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    labels, edges = read_graph(DATA / "network-ho.csv")
    order = read_order(GLSN / "critical/DND-DRL/results/formal_removal_sequence.csv", labels)
    states = HistoricalStates(labels)
    index = {label: i for i, label in enumerate(labels)}
    original = set(edges)
    incoming, outgoing = neighbors(len(labels), original)
    edge_groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in csv_rows("selected_edges.csv"):
        edge_groups[row["port"], row["method"], int(row["repeat"])].append(row)
    for (port, method, repeat), group in edge_groups.items():
        added = set()
        extra_in: dict[int, set[int]] = defaultdict(set)
        extra_out: dict[int, set[int]] = defaultdict(set)
        last_rank = 0
        for number, row in enumerate(group, start=1):
            rank = int(row["priority_rank"])
            critical = order[rank - 1]
            source = index[row["source_state"]]
            target = index[row["target_state"]]
            edge = source, target
            assert int(row["edge_number"]) == number
            assert states.ports[critical] == port
            assert row["critical_state"] == labels[critical] and rank > last_rank
            assert source != target and target != critical
            assert edge not in original and edge not in added
            assert source in incoming[critical] | extra_in[critical]
            assert any(states.update(source, states.ports[w]) == target
                       for w in outgoing[critical] | extra_out[critical] if w != critical)
            score = (len(outgoing[source]) + len(extra_out[source]) +
                     len(incoming[target]) + len(extra_in[target]))
            assert score == int(row["degree_at_choice"])
            added.add(edge)
            extra_out[source].add(target)
            extra_in[target].add(source)
            last_rank = rank

    curves: dict[tuple[str, str, int, str, int], list[dict]] = defaultdict(list)
    for row in csv_rows("port_curves.csv"):
        key = (row["port"], row["method"], int(row["repeat"]),
               row["scenario"], int(row["budget"]))
        curves[key].append(row)
    trial_groups: dict[tuple[str, str, int, str], list[dict]] = defaultdict(list)
    trials = csv_rows("port_trials.csv")
    for row in trials:
        port, method, repeat = row["port"], row["method"], int(row["repeat"])
        scenario, budget = row["scenario"], int(row["budget"])
        trial_key = port, method, repeat, scenario, budget
        before_key = port, "Baseline", 0, scenario, 0
        assert len(edge_groups[port, method, repeat]) >= budget
        assert len(curves[trial_key]) == len(curves[before_key]) == 17
        q = [int(point["attacked"]) for point in curves[trial_key]]
        before_q = [int(point["attacked"]) for point in curves[before_key]]
        assert q == before_q and q[0] == 0 and q[-1] == 2144
        after = area([float(point["fraction_in_gscc"]) for point in curves[trial_key]],
                     q, len(labels))
        before = area([float(point["fraction_in_gscc"]) for point in curves[before_key]],
                      q, len(labels))
        assert math.isclose(before, float(row["r_before"]), abs_tol=1e-12)
        assert math.isclose(after, float(row["r_after"]), abs_tol=1e-12)
        assert math.isclose(after - before, float(row["delta_a"]), abs_tol=1e-12)
        assert math.isclose((after - before) / before, float(row["delta_r"]),
                            abs_tol=1e-12)
        trial_groups[port, method, budget, scenario].append(row)

    summaries = csv_rows("port_summary.csv")
    for row in summaries:
        group = trial_groups[row["port"], row["method"], int(row["budget"]),
                             row["scenario"]]
        assert len(group) == int(row["repeats"])
        assert math.isclose(sum(float(x["delta_r"]) for x in group) / len(group),
                            float(row["mean_delta_r"]), abs_tol=1e-12)
    print(f"OK: {len(edge_groups)} port-method runs, {len(trials)} trial outcomes, "
          f"{len(summaries)} port summaries")


if __name__ == "__main__":
    main()
