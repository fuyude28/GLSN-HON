"""Replay new ranking-guided edges and verify saved curves and summaries."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict

from ranking_ldf_experiment import (
    GLSN, HERE, METHODS, RANKING_SOURCES, load_rankings, read_csv,
)
from robustness_experiment import (
    DATA, HistoricalStates, attack_schedule, bypass_candidates, evaluate, neighbors, read_graph,
)


def main() -> None:
    results = HERE / "results/ranking_ldf"
    protocol = json.loads((results / "protocol.json").read_text(encoding="utf-8"))
    labels, base_edges = read_graph(DATA / "network-ho.csv")
    n = len(labels)
    index = {label: node for node, label in enumerate(labels)}
    orders, sources = load_rankings(labels)
    saved_lengths = {row["method"]: row["saved_ranked_states"] for row in sources}
    edge_groups = defaultdict(list)
    for row in read_csv(results / "selected_edges.csv"):
        edge_groups[row["method"], int(row["repeat"])].append(row)
    seeds, budgets = protocol["random_seeds"], protocol["budgets"]
    expected = {(method, 0) for method in RANKING_SOURCES}
    expected.update(("RandomNode", repeat) for repeat in range(len(seeds)))
    assert set(edge_groups) == expected
    curve_groups = defaultdict(list)
    for row in read_csv(results / "gscc_curves.csv"):
        curve_groups[row["method"], int(row["repeat"]), int(row["budget"])].append(row)
    assert set(curve_groups) == {(method, repeat, b) for method, repeat in expected for b in budgets}
    attacked, active = attack_schedule(n, orders["DND-DRL"], protocol["attack_rounds"])
    states = HistoricalStates(labels)
    verified_curves = 0
    sample_random_repeats = {0, len(seeds) // 2, len(seeds) - 1}
    sample_budgets = {budgets[0], budgets[(len(budgets) - 1) // 2], budgets[-1]}
    for (method, repeat), trace in edge_groups.items():
        seed = seeds[repeat] if method == "RandomNode" else seeds[0]
        order = random.Random(seed).sample(range(n), n) if method == "RandomNode" else orders[method]
        saved_length = n if method == "RandomNode" else saved_lengths[method]
        existing = set(base_edges)
        incoming, outgoing = neighbors(n, existing)
        edges, previous_rank = [], 0
        assert len(trace) == max(budgets)
        for number, row in enumerate(trace, start=1):
            rank = int(row["priority_rank"])
            assert int(row["seed"]) == seed and int(row["edge_number"]) == number
            assert previous_rank < rank and labels[order[rank - 1]] == row["critical_state"]
            assert (row["from_unranked_tail"] == "True") == (rank > saved_length)
            for skipped in order[previous_rank:rank - 1]:
                assert not bypass_candidates(skipped, states, incoming, outgoing, existing)
            candidates = bypass_candidates(order[rank - 1], states, incoming, outgoing, existing)
            u, v = index[row["source_state"]], index[row["target_state"]]
            chosen = min(candidates, key=lambda e: (len(outgoing[e[0]]) + len(incoming[e[1]]), e))
            assert (u, v) == chosen, (method, repeat, number)
            assert int(row["directed_degree_at_choice"]) == len(outgoing[u]) + len(incoming[v])
            existing.add(chosen)
            outgoing[u].add(v)
            incoming[v].add(u)
            edges.append(chosen)
            previous_rank = rank
        if method != "RandomNode" or repeat in sample_random_repeats:
            for budget in sample_budgets:
                _, actual = evaluate(n, base_edges + edges[:budget], attacked, active)
                saved = [int(row["gscc"]) for row in curve_groups[method, repeat, budget]]
                assert actual == saved, (method, repeat, budget)
                verified_curves += 1
    baseline = read_csv(results / "baseline_curve.csv")
    trials = read_csv(results / "trials.csv")
    assert len(trials) == len(curve_groups)
    grouped, seen = defaultdict(list), set()
    for row in trials:
        key = row["method"], int(row["repeat"]), int(row["budget"])
        assert key not in seen
        seen.add(key)
        points = curve_groups[key]
        q, g = [int(p["attacked"]) for p in points], [int(p["gscc"]) for p in points]
        assert q == attacked
        assert all(size >= int(base["gscc"]) for size, base in zip(g, baseline))
        area = sum((a + b) * (q1 - q0) for a, b, q0, q1 in
                   zip(g[:-1], g[1:], q[:-1], q[1:])) / (2 * n * n)
        assert math.isclose(area, float(row["auc"]), rel_tol=0, abs_tol=1e-12)
        assert math.isclose(float(row["baseline_auc"]), protocol["baseline_partial_auc"], rel_tol=0, abs_tol=1e-12)
        delta = (area - protocol["baseline_partial_auc"]) / protocol["baseline_partial_auc"]
        assert math.isclose(delta, float(row["delta_r"]), rel_tol=0, abs_tol=1e-12)
        assert int(row["edges_added"]) == key[2] and int(row["final_gscc"]) == g[-1]
        grouped[row["method"], key[2]].append(delta)
    summary = read_csv(results / "summary.csv")
    assert len(summary) == len(METHODS) * len(budgets)
    for row in summary:
        values = grouped[row["method"], int(row["budget"])]
        assert int(row["trials"]) == len(values)
        assert int(row["min_edges_added"]) == int(row["budget"])
        assert math.isclose(statistics.mean(values), float(row["mean_delta_r"]), rel_tol=0, abs_tol=1e-12)
        if len(values) > 1:
            assert math.isclose(statistics.stdev(values), float(row["sd_delta_r"]), rel_tol=0, abs_tol=1e-12)
        else:
            assert row["sd_delta_r"] == ""
    print(f"OK: {len(edge_groups)} runs; {sum(map(len, edge_groups.values()))} CB-LDF edges replayed; "
          f"{len(trials)} AUCs; {verified_curves} independently reevaluated GSCC curves; "
          f"{len(summary)} summary rows")


if __name__ == "__main__":
    main()
