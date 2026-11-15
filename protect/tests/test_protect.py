import igraph as ig
import pytest

import protect_experiment as experiment
from protection_rankings import DirectedGraph


def complete(n):
    return DirectedGraph.from_edges(n, [(u,v) for u in range(n) for v in range(n) if u != v])


class Rankings:
    def rank(self, method, live, removed, effective):
        if method == 'changing' and removed:
            return sorted(effective, reverse=True)
        return sorted(effective)


def test_previous_round_protection_expires_and_formerly_protected_nodes_can_fail():
    curve, rounds, end = experiment.run_trial(complete(6), Rankings(), 'changing', 'targeted', 42, 2)
    assert rounds[0]['protected_nodes'] == '[0, 1]'
    assert rounds[0]['removed_nodes'] == '[2, 3]'
    assert rounds[1]['protected_nodes'] == '[4, 5]'
    assert rounds[1]['removed_nodes'] == '[0, 1]'
    assert end['terminal_gscc'] == 2
    assert end['stop_reason'] == 'gscc_candidates_at_most_batch'
    assert max(row['protected'] for row in curve) == 2


def test_random_attack_filters_protected_candidates_without_refilling(monkeypatch):
    monkeypatch.setattr(experiment, 'random_candidates', lambda *args: [0, 2])
    curve, rounds, end = experiment.run_trial(complete(3), Rankings(), 'fixed', 'random', 42, 2)
    assert rounds[0]['attack_candidates'] == '[0, 2]'
    assert rounds[0]['removed_nodes'] == '[2]'
    assert end['attacks'] == 1
    assert end['terminal_gscc'] == 2


def test_a_fully_blocked_random_batch_does_not_terminate_the_experiment(monkeypatch):
    draws = iter([[0, 1], [0, 2]])
    monkeypatch.setattr(experiment, 'random_candidates', lambda *args: next(draws))
    curve, rounds, end = experiment.run_trial(complete(3), Rankings(), 'fixed', 'random', 42, 2)
    assert rounds[0]['removed_nodes'] == '[]'
    assert end['rounds'] == 2
    assert end['attacks'] == 1


def test_stop_at_gscc_collapse_in_the_middle_of_an_attack_batch(monkeypatch):
    graph = DirectedGraph.from_edges(5, [(i, (i+1)%5) for i in range(5)])
    monkeypatch.setattr(experiment, 'random_candidates', lambda *args: [0, 1])
    curve, rounds, end = experiment.run_trial(graph, Rankings(), 'No protection', 'random', 42, 2)
    assert end['attacks'] == 1
    assert end['terminal_gscc'] == 1
    assert end['stop_reason'] == 'gscc_collapsed'
    assert rounds[0]['removed_nodes'] == '[0]'


def test_audc_is_exact_post_removal_sum_not_trapezoidal_batch_approximation():
    curve, rounds, end = experiment.run_trial(complete(6), Rankings(), 'No protection', 'targeted', 42, 1)
    assert [row['gscc'] for row in curve] == [6, 5, 4, 3, 2, 1]
    assert end['audc'] == pytest.approx(15/36)
    assert curve[-1]['cumulative_audc'] == end['audc']


def test_no_attack_when_initial_gscc_pool_is_at_most_one_batch():
    curve, rounds, end = experiment.run_trial(complete(3), Rankings(), 'fixed', 'targeted', 42, 3)
    assert rounds == []
    assert end['attacks'] == 0
    assert end['terminal_gscc'] == 3
    assert end['stop_reason'] == 'gscc_candidates_at_most_batch'


def test_shared_random_priorities_remain_uniform_candidates_before_protection():
    import random
    first = experiment.random_candidates(random.Random(42), 10, [1, 3, 6, 9], 3)
    second = experiment.random_candidates(random.Random(42), 10, [1, 3, 6, 9], 3)
    assert first == second
    assert len(set(first)) == 3
    assert set(first) <= {1, 3, 6, 9}


@pytest.mark.parametrize('removed', [[], [0], [0, 3], [0, 1, 3]])
def test_gscc_pool_matches_dismantling_including_ties_and_collapse(removed):
    graph = DirectedGraph.from_edges(8, [(0,1),(1,0),(1,2),(2,1),(2,0),(0,2),
                                        (3,4),(4,3),(2,5),(5,6),(6,7)])
    live = ig.Graph(n=8, edges=graph.residual_edges(removed), directed=True)
    assert experiment.gscc_candidates(live) == graph.gscc_candidates(removed)


def test_external_high_degree_node_is_excluded_from_protection_and_random_attack(monkeypatch):
    from protection_rankings import ProtectionRankings
    graph = DirectedGraph.from_edges(6, [(i,(i+1)%5) for i in range(5)] +
                                    [(5,i) for i in range(5)])
    rankings = ProtectionRankings(graph, ['Dynamic out-degree'])
    seen = []
    def draw(rng, n, pool, batch):
        seen.append(pool)
        return [2,3]
    monkeypatch.setattr(experiment, 'random_candidates', draw)
    curve, rounds, end = experiment.run_trial(graph, rankings, 'Dynamic out-degree', 'random', 42, 2)
    assert seen == [[0,1,2,3,4]]
    assert rounds[0]['protected_nodes'] == '[0, 1]'
    assert rounds[0]['gscc_candidates_before'] == 5
    assert end['terminal_gscc'] == 1


def test_real_dnd_first_protection_batch_matches_formal_dismantling():
    import csv
    import torch
    from protection_rankings import CRITICAL, ProtectionRankings
    torch.set_num_threads(1)
    graph = DirectedGraph.read_csv_edge_table(CRITICAL.parent.parent / 'data/network-ho.csv')
    live = ig.Graph(n=graph.num_nodes, edges=graph.edges, directed=True)
    pool = experiment.gscc_candidates(live)
    rankings = ProtectionRankings(graph, ['DND-DRL'])
    selected = rankings.rank('DND-DRL', live, set(), pool)[:134]
    with (CRITICAL / 'DND-DRL/results/formal_removal_sequence.csv').open(
            encoding='utf-8', newline='') as handle:
        formal = [graph.node_mapping[row['state_node']] for row in csv.DictReader(handle)][:134]
    assert set(selected) == set(formal)
    assert set(selected) <= set(pool)
