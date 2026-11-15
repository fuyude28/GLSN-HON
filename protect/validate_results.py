"""Verify exported Gu-style trials against their round logs and residual graph."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import igraph as ig

from protection_rankings import DirectedGraph

HERE = Path(__file__).resolve().parent


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def validate(results: Path, graph_path: Path) -> dict:
    meta = json.loads((results / 'protection_summary.json').read_text(encoding='utf-8'))
    graph = DirectedGraph.read_csv_edge_table(graph_path)
    topology = ig.Graph(n=graph.num_nodes, edges=graph.edges, directed=True)
    neighbors = [set(topology.neighbors(node, mode='all')) for node in range(graph.num_nodes)]
    trials = read_rows(results / 'protection_trials.csv')
    expected = {(mode, method, seed) for mode in ('targeted','random')
                for method in meta['protection_methods']
                for seed in (meta['random_seeds'] if mode == 'random' or method == 'Random protection'
                             else meta['random_seeds'][:1])}
    actual = {(row['mode'], row['method'], int(row['seed'])) for row in trials}
    assert actual == expected and len(trials) == len(expected), 'trial/seed coverage mismatch'
    checked_points = 0
    previously_protected_failures = 0
    gscc_rounds_checked = 0
    for trial in trials:
        prefix = trial['run_prefix']
        curve = read_rows(results / 'runs' / (prefix + '_curve.csv'))
        logs_path = results / 'runs' / (prefix + '_rounds.csv')
        rounds = read_rows(logs_path) if int(trial['rounds']) else []
        removed = set()
        history = set()
        sequence = []
        live = topology.copy()
        for log in rounds:
            # Reconstruct the round-start action mask independently from
            # the recorded protection and attack selections.
            components = live.connected_components(mode='strong')
            component_sizes = components.sizes()
            largest = max(component_sizes)
            pool = {node for node, cid in enumerate(components.membership)
                    if component_sizes[cid] == largest} if largest > 1 else set()
            assert len(pool) > meta['batch_size']
            assert len(pool) == int(log['gscc_candidates_before'])
            assert largest == int(log['gscc_before'])
            protected = set(json.loads(log['protected_nodes']))
            candidates = json.loads(log['attack_candidates'])
            applied = json.loads(log['removed_nodes'])
            assert len(protected) == (0 if trial['method'] == 'No protection' else meta['batch_size'])
            assert not protected & removed
            assert protected <= pool
            assert set(candidates) <= pool
            assert all(neighbors[node] - removed for node in protected)
            assert all(node not in removed and neighbors[node] - removed for node in candidates)
            assert len(candidates) <= meta['batch_size']
            if trial['mode'] == 'random':
                assert len(candidates) == meta['batch_size']
            else:
                assert not set(candidates) & protected
            allowed = [node for node in candidates if node not in protected]
            assert applied == allowed[:len(applied)]
            if len(applied) < len(allowed):
                assert int(log['gscc_after']) <= 1
            assert not set(applied) & (protected | removed)
            assert len(set(applied)) == len(applied)
            incident_edges = {edge for node in applied for edge in live.incident(node, mode='all')}
            live.delete_edges(list(incident_edges))
            gscc_rounds_checked += 1
            previously_protected_failures += len(set(applied) & history)
            removed.update(applied)
            history.update(protected)
            sequence.extend(applied)
            assert len(removed) == int(log['attacked_after'])
        assert sequence == [int(row['removed_node']) for row in curve[1:]]
        assert len(curve) == int(trial['attacks']) + 1
        assert [int(row['attacked']) for row in curve] == list(range(len(curve)))
        sizes = [int(row['gscc']) for row in curve]
        assert all(a >= b for a,b in zip(sizes,sizes[1:]))
        audc = math.fsum(sizes[1:]) / (meta['nodes'] * meta['initial_gscc'])
        assert math.isclose(audc, float(trial['audc']), abs_tol=1e-12)
        remaining = [node for node in range(graph.num_nodes) if node not in removed]
        residual = topology.induced_subgraph(remaining)
        final_size = max(residual.connected_components(mode='strong').sizes(), default=0)
        effective = sum(degree > 0 for degree in residual.degree(mode='all'))
        assert final_size == sizes[-1] == int(trial['terminal_gscc'])
        assert effective == int(trial['terminal_effective_nodes'])
        final_components = residual.connected_components(mode='strong').sizes()
        candidate_count = sum(value for value in final_components if value == final_size) if final_size > 1 else 0
        assert candidate_count == int(trial['terminal_gscc_candidates'])
        if trial['stop_reason'] == 'gscc_collapsed':
            assert final_size <= 1
        else:
            assert trial['stop_reason'] == 'gscc_candidates_at_most_batch'
            assert final_size > 1 and candidate_count <= meta['batch_size']
        checked_points += len(curve)
    report = dict(trials=len(trials), per_removal_points=checked_points,
                  seed_coverage='passed', per_round_protection_and_attack='passed',
                  exact_audc='passed', independently_reconstructed_terminal_graphs='passed',
                  gscc_only_rounds_verified=gscc_rounds_checked,
                  previously_protected_nodes_later_attacked=previously_protected_failures)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'results')
    parser.add_argument('--graph', type=Path, default=HERE.parent.parent / 'data/network-ho.csv')
    args = parser.parse_args()
    report = validate(args.results, args.graph)
    (args.results / 'verification_gu.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
