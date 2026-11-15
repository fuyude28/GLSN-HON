"""Sparse implementation of the TAD back-edge ranking in Code/TAD_analysis.py."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lsmr


def tad_order(num_nodes: int, edges: tuple[tuple[int, int], ...]
              ) -> tuple[list[int], dict[str, int | float]]:
    """Rank nodes by the summed incident trophic back-edge contributions.

    Matches TAD_analysis.get_levels and its `up` score, without constructing
    the N x N adjacency, trophic-difference, or back-edge matrices.
    """
    pairs = np.asarray([(u, v) for u, v in edges if u != v], dtype=np.int64)
    if len(pairs) == 0:
        return list(range(num_nodes)), {"back_edges": 0, "lsmr_iterations": 0,
                                        "lsmr_stop": 0}
    source, target = pairs[:, 0], pairs[:, 1]
    adjacency = coo_matrix((np.ones(len(pairs)), (source, target)),
                           shape=(num_nodes, num_nodes)).tocsr()
    indegree = np.asarray(adjacency.sum(axis=0)).ravel()
    outdegree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = diags(indegree + outdegree) - adjacency - adjacency.T
    solution = lsmr(laplacian, indegree - outdegree)
    levels = solution[0]

    delta = levels[target] - levels[source]
    contributions = np.where(delta < 0, (delta - 1) ** 2, 0.0)
    scores = np.zeros(num_nodes)
    np.add.at(scores, source, contributions)
    np.add.at(scores, target, contributions)
    order = np.lexsort((np.arange(num_nodes), -scores)).tolist()
    return order, {"back_edges": int(np.count_nonzero(delta < 0)),
                   "lsmr_iterations": int(solution[2]),
                   "lsmr_stop": int(solution[1]),
                   "lsmr_residual": float(solution[3])}
