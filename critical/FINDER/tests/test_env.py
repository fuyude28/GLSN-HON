import math

from finder_pytorch.env import NetworkDismantlingEnv
from finder_pytorch.graph import FinderGraph


def test_env_reward_and_terminal_match_nd_objective():
    graph = FinderGraph.from_edges(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])
    env = NetworkDismantlingEnv(graph)

    assert env.candidate_actions() == [0, 1, 2, 3]
    reward = env.step(1)

    assert math.isclose(reward, -2 / 16)
    assert env.removed == [1]
    assert not env.is_terminal()
    assert env.candidate_actions() == [2, 3]

    second_reward = env.step(2)

    assert math.isclose(second_reward, -1 / 16)
    assert env.is_terminal()
    assert env.candidate_actions() == []


def test_env_robustness_uses_complete_solution_order():
    graph = FinderGraph.from_edges(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])

    assert math.isclose(NetworkDismantlingEnv.robustness(graph, [1, 2, 0, 3]), 4 / 16)
