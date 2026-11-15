from finder_pytorch.evaluate import summarize_solution
from finder_pytorch.graph import FinderGraph


def test_summarize_solution_reports_nd_metrics():
    graph = FinderGraph.from_edges(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])

    summary = summarize_solution(graph, [1, 2])

    assert summary["num_nodes"] == 4
    assert summary["num_edges"] == 3
    assert summary["removed_until_terminal"] == 2
    assert summary["robustness"] == 0.25
    assert summary["initial_lcc"] == 4
