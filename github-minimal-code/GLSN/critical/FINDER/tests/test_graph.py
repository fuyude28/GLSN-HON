import networkx as nx

from finder_pytorch.graph import FinderGraph


def test_finder_graph_normalizes_edges_and_candidates():
    graph = FinderGraph.from_edges(num_nodes=5, edges=[(0, 1), (1, 0), (1, 2), (3, 4)])

    assert graph.num_nodes == 5
    assert graph.edges == ((0, 1), (1, 2), (3, 4))
    assert graph.active_nodes([]) == [0, 1, 2, 3, 4]
    assert graph.candidate_nodes([]) == [0, 1, 2, 3, 4]
    assert graph.candidate_nodes([1]) == [3, 4]
    assert graph.largest_component_size([]) == 3
    assert graph.largest_component_size([1]) == 2


def test_finder_graph_from_networkx_relabels_to_integers():
    nx_graph = nx.Graph()
    nx_graph.add_edges_from([("a", "b"), ("b", "c")])

    graph = FinderGraph.from_networkx(nx_graph)

    assert graph.num_nodes == 3
    assert set(graph.edges) == {(0, 1), (1, 2)}
    assert graph.original_node(0) in {"a", "b", "c"}
    assert graph.internal_node(graph.original_node(0)) == 0


def test_finder_graph_reads_weighted_csv_edge_table(tmp_path):
    csv_path = tmp_path / "state_edges.csv"
    csv_path.write_text(
        "node1,node2,weight\n"
        "1-1,2-1,0.5\n"
        "2-1,3-2,1.0\n"
        "3-2,1-1,0.2\n",
        encoding="utf-8",
    )

    graph = FinderGraph.read_csv_edge_table(csv_path, source_col="node1", target_col="node2")

    assert graph.num_nodes == 3
    assert len(graph.edges) == 3
    assert graph.internal_node("1-1") != graph.internal_node("2-1")
    assert graph.original_node(graph.internal_node("3-2")) == "3-2"
