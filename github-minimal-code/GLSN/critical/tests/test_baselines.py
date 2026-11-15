import csv
import io
import random

import pytest

from run_baselines import CURVE_FIELDS, dismantle, rank_candidates
from dnd_drl.graph import DirectedGraph


def test_directed_degree_rankings_are_distinct():
    graph = DirectedGraph.from_edges(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    candidates = graph.gscc_candidates()
    assert rank_candidates(graph, set(), candidates, "dynamic_in_degree",
                           random.Random(1), 1) == [0]
    assert rank_candidates(graph, set(), candidates, "dynamic_out_degree",
                           random.Random(1), 1) == [2]
    assert rank_candidates(graph, set(), candidates, "dynamic_total_degree",
                           random.Random(1), 1) == [2]
    assert len(rank_candidates(graph, set(), candidates, "dynamic_pagerank",
                               random.Random(1), 2)) == 2


def test_baseline_curve_has_exact_per_removal_auc():
    graph = DirectedGraph.from_edges(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CURVE_FIELDS)
    writer.writeheader()
    summary = dismantle(graph, "dynamic_total_degree", 42, 0.25, writer,
                        log_interval=0)
    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert summary["initial_gscc"] == 3
    assert summary["removals"] == 1
    assert summary["auc"] == pytest.approx(1 / 16)
    assert [int(row["gscc_size"]) for row in rows] == [3, 1]
    assert float(rows[-1]["cumulative_auc"]) == pytest.approx(summary["auc"])
