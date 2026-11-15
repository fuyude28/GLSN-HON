import numpy as np
from scipy.sparse.linalg import lsmr

from run_structural import evaluate
from corehd_dismantle import decycle
from corehd_ho import corehd_order
from tad_sparse import tad_order
from dnd_drl.graph import DirectedGraph


def test_corehd_one_at_a_time_matches_reference_decycling():
    edges = ((0, 1), (1, 2), (2, 0), (2, 3), (3, 4),
             (4, 2), (4, 5), (5, 6), (6, 4))
    adjacency = {node: {} for node in range(7)}
    for u, v in edges:
        adjacency[u][v] = adjacency[v][u] = 1.0
    expected = decycle(adjacency, step_fraction=1 / 7)
    order = corehd_order(7, edges)
    assert order[:len(expected)] == expected
    assert sorted(order) == list(range(7))


def test_sparse_tad_scores_rank_like_original_dense_formula():
    edges = ((0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 1))
    n = 5
    adjacency = np.zeros((n, n))
    for u, v in edges:
        adjacency[u, v] = 1
    win = adjacency.sum(axis=0)
    wout = adjacency.sum(axis=1)
    levels = lsmr(np.diag(win + wout) - adjacency - adjacency.T,
                  win - wout)[0]
    delta = levels[None, :] - levels[:, None]
    contribution = (np.where(delta >= 0, 1, delta) - 1) ** 2 * adjacency
    expected = np.lexsort((np.arange(n),
                           -(contribution.sum(axis=0) + contribution.sum(axis=1))))
    order, details = tad_order(n, edges)
    assert order == expected.tolist()
    assert details["back_edges"] == int(np.count_nonzero((delta < 0) & (adjacency > 0)))


def test_structural_curve_stops_at_gscc_one(tmp_path):
    graph = DirectedGraph.from_edges(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    summary = evaluate(graph, [2, 0, 1, 3], tmp_path / "curve.csv", 0.25)
    assert summary["removals"] == 1
    assert summary["auc"] == 1 / 16
