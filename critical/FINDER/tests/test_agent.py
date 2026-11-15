import torch

from finder_pytorch.agent import FinderDQNAgent
from finder_pytorch.env import NetworkDismantlingEnv
from finder_pytorch.graph import FinderGraph
from finder_pytorch.replay import NStepReplayBuffer


def test_agent_optimization_step_updates_from_replay():
    graph = FinderGraph.from_edges(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])
    env = NetworkDismantlingEnv(graph)
    env.step(1)
    env.step(2)

    replay = NStepReplayBuffer(capacity=10)
    replay.add_episode(env.episode, n_step=2)
    agent = FinderDQNAgent(hidden_dim=16, batch_size=2, gamma=1.0)

    loss = agent.optimize(replay)

    assert isinstance(loss, float)
    assert loss >= 0


def test_agent_greedy_solution_stops_when_residual_edges_are_gone():
    torch.manual_seed(0)
    graph = FinderGraph.from_edges(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])
    agent = FinderDQNAgent(hidden_dim=16, batch_size=2, gamma=1.0)

    solution = agent.greedy_solution(graph, step_size=1)

    assert 1 <= len(solution) <= 3
    assert len(solution) == len(set(solution))


def test_agent_greedy_solution_reports_progress():
    graph = FinderGraph.from_edges(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])
    agent = FinderDQNAgent(hidden_dim=16, batch_size=2, gamma=1.0)
    rows: list[dict[str, float | int]] = []

    solution = agent.greedy_solution(graph, step_size=1, log_interval=1, progress_callback=rows.append)

    assert 1 <= len(solution) <= 3
    assert len(rows) >= 2
    assert rows[0]["step"] == 0
    assert rows[-1]["step"] == len(solution)
    assert rows[-1]["residual_edges"] == 0


def test_agent_batch_greedy_allows_actions_that_become_isolated_inside_batch():
    graph = FinderGraph.from_edges(num_nodes=4, edges=[(0, 1), (2, 3)])
    agent = FinderDQNAgent(hidden_dim=16, batch_size=2, gamma=1.0)

    class FixedRanking:
        def q_values(self, graph, removed):
            return torch.tensor([10.0, 9.0, 8.0, 7.0], device=agent.device)

    agent.online = FixedRanking()
    solution = agent.greedy_solution(graph, step_size=3)

    assert solution[:3] == [0, 1, 2]
    assert len(solution) == len(set(solution))


def test_agent_greedy_solution_supports_fractional_batches():
    graph = FinderGraph.from_edges(num_nodes=10, edges=[(node, node + 1) for node in range(9)])
    agent = FinderDQNAgent(hidden_dim=16, batch_size=2, gamma=1.0)
    rows: list[dict[str, float | int]] = []

    solution = agent.greedy_solution(
        graph,
        step_fraction=0.1,
        log_interval=1,
        progress_callback=rows.append,
    )

    assert len(solution) == len(set(solution))
    assert rows[0]["step"] == 0
    assert rows[1]["step"] == 1
    assert rows[-1]["residual_edges"] == 0
