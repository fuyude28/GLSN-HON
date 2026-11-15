import torch

from finder_pytorch.graph import FinderGraph
from finder_pytorch.model import FinderQNetwork


def test_model_returns_one_q_value_per_node_and_masks_removed_nodes():
    graph = FinderGraph.from_edges(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])
    model = FinderQNetwork(hidden_dim=16, message_passing_steps=2)

    q_values = model.q_values(graph, removed=[1])

    assert q_values.shape == (4,)
    assert torch.isfinite(q_values[[0, 2, 3]]).all()
    assert q_values[1].item() < -1e20


def test_model_can_score_selected_actions_as_batch():
    graph = FinderGraph.from_edges(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])
    model = FinderQNetwork(hidden_dim=16, message_passing_steps=2)

    q_values = model.score_actions([graph, graph], removed_batch=[[], [1]], actions=[1, 2])

    assert q_values.shape == (2,)
    assert torch.isfinite(q_values).all()


def test_sparse_adjacency_keeps_original_message_passing_values():
    graph = FinderGraph.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    model = FinderQNetwork(hidden_dim=16, message_passing_steps=2)
    adjacency = model.adjacency_matrix(graph, {1}, torch.device("cpu"))
    expected = torch.zeros(4, 4)
    expected[2, 3] = expected[3, 2] = 1.0
    assert adjacency.is_sparse
    assert torch.equal(adjacency.to_dense(), expected)
