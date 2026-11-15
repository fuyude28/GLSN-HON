"""Exact supervised labels from all minimum LSCC-disintegration sets."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import igraph as ig
import numpy as np


@dataclass(frozen=True)
class LabelResult:
    probabilities: np.ndarray
    min_removals: int
    optimal_set_count: int
    combinations_checked: int
    target_lscc_size: int


def minimum_set_labels(graph: ig.Graph, target_fraction: float = 0.1,
                       max_combinations: int | None = None) -> LabelResult:
    """Enumerate minimum-cardinality removal sets and average membership.

    The article's target LSCC/N <= 0.1 gives floor(0.1*N) for integer LSCC.
    Raise rather than silently replacing incomplete exact labels with a
    heuristic when a computational budget is reached.
    """
    if not graph.is_directed():
        raise ValueError("DND labels require a directed graph")
    n = graph.vcount()
    target = math.floor(target_fraction * n)
    checked = 0
    for removed_count in range(n + 1):
        membership = np.zeros(n, dtype=np.float64)
        optimum_count = 0
        for removed in itertools.combinations(range(n), removed_count):
            checked += 1
            if max_combinations is not None and checked > max_combinations:
                raise RuntimeError(f"exact labeling exceeded {max_combinations} combinations")
            survivor = graph.copy()
            survivor.delete_vertices(removed)
            largest = max(survivor.connected_components(mode="strong").sizes(),
                          default=0)
            if largest <= target:
                optimum_count += 1
                membership[list(removed)] += 1
        if optimum_count:
            return LabelResult(membership / optimum_count, removed_count,
                               optimum_count, checked, target)
    raise RuntimeError("no disintegration set found")
