import random
from pathlib import Path

import pytest
import torch

from dnd_drl.agent import AgentConfig, DNDDrLAgent
from dnd_drl.env import DirectedDismantlingEnv
from dnd_drl.graph import DirectedGraph, directed_ba_graph
from dnd_drl.model import DirQNetwork, directed_neighbor_means, normalized_residual_degrees
from dnd_drl.replay import NStepReplayBuffer
from dnd_drl.training import (TrainingConfig, load_training_checkpoint,
                              save_training_checkpoint, train)


def test_graph_preserves_direction_and_gscc(tmp_path):
    path = tmp_path / "state_edges.csv"
    path.write_text("FromNode,ToNode\na|,b|\nb|,c|\nc|,a|\nc|,d|\n", encoding="utf-8")
    graph = DirectedGraph.read_csv_edge_table(path)
    assert graph.num_nodes == 4
    assert len(graph.edges) == 4
    assert graph.gscc_size() == 3
    assert graph.gscc_size({graph.node_mapping["a|"]}) == 1
    assert (graph.node_mapping["b|"], graph.node_mapping["a|"]) not in graph.edges


def test_reward_auc_and_terminal_are_gscc_based():
    graph = DirectedGraph.from_edges(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    env = DirectedDismantlingEnv(graph)
    assert env.gscc_size == 3
    assert env.candidate_actions() == [0, 1, 2]
    with pytest.raises(ValueError, match="largest SCC"):
        env.step(3)
    assert env.step(0) == pytest.approx(-1 / 4)
    assert env.is_terminal()
    assert -sum(env.episode.rewards) / graph.num_nodes == pytest.approx(1 / 16)
    assert env.episode.terminals == [True]


def test_tied_largest_sccs_are_both_actionable():
    graph = DirectedGraph.from_edges(5, [(0, 1), (1, 0), (2, 3), (3, 2), (3, 4)])
    env = DirectedDismantlingEnv(graph)
    assert env.candidate_actions() == [0, 1, 2, 3]
    env.step(0)
    assert env.candidate_actions() == [2, 3]


def test_preselected_batch_keeps_original_ranking_after_gscc_changes():
    graph = DirectedGraph.from_edges(5, [(0, 1), (1, 2), (2, 0),
                                         (3, 4), (4, 3)])
    env = DirectedDismantlingEnv(graph)
    assert env.candidate_actions() == [0, 1, 2]
    assert env.remove_batch([0, 1]) == [(1, 2), (2, 2)]
    assert env.removed == [0, 1]
    assert env.candidate_actions() == [3, 4]


def test_inference_batches_one_percent_of_all_remaining_states():
    n = 101
    graph = DirectedGraph.from_edges(
        n, [(node, (node + 1) % n) for node in range(n)] +
        [((node + 1) % n, node) for node in range(n)])
    agent = DNDDrLAgent(AgentConfig(hidden_dim=8, message_passing_steps=1,
                                    device="cpu"))

    class FixedRanking(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, graph, removed=()):
            self.calls += 1
            return -torch.arange(graph.num_nodes, dtype=torch.float32)

    agent.online = FixedRanking()
    rounds = []
    curve = []
    sequence = agent.greedy_solution(
        graph, batch_fraction=0.01, max_removals=3,
        round_progress=lambda round_no, size: rounds.append((round_no, size)),
        progress=lambda step, size: curve.append((step, size)))
    assert sequence == [0, 1, 2]
    assert rounds == [(1, 2), (2, 1)]  # ceil(101 * 1%) then ceil(99 * 1%)
    assert agent.online.calls == 2
    assert curve == [(0, 101), (1, 100), (2, 99), (3, 98)]
    assert sum(size for _, size in curve[1:]) / n**2 == pytest.approx(297 / n**2)


def test_greedy_and_random_actions_exclude_nodes_outside_gscc():
    graph = DirectedGraph.from_edges(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    agent = DNDDrLAgent(AgentConfig(hidden_dim=8, message_passing_steps=1, device="cpu"))
    agent.online.forward = lambda graph, removed=(): torch.tensor([0., 1., 2., 100.])
    assert agent.select_action(graph, [], epsilon=0, rng=random.Random(1)) == 2
    assert all(agent.select_action(graph, [], epsilon=1, rng=random.Random(seed)) != 3
               for seed in range(20))


def test_incoming_and_outgoing_means_are_separate():
    h = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
    sources = torch.tensor([0, 2, 1])
    targets = torch.tensor([1, 1, 2])
    mean_in, mean_out, degree_in, degree_out = directed_neighbor_means(h, sources, targets)
    assert mean_in[:, 0].tolist() == pytest.approx([0.0, 2.5, 2.0, 0.0])
    assert mean_out[:, 0].tolist() == pytest.approx([2.0, 4.0, 2.0, 0.0])
    assert degree_in.tolist() == [0, 2, 1, 0]
    assert degree_out.tolist() == [1, 1, 1, 0]


def test_degree_normalization_does_not_shrink_with_node_count():
    degree = torch.tensor([[1.0, 2.0], [2.0, 1.0], [0.0, 0.0]])
    padded = torch.cat([degree, torch.zeros((1000, 2))])
    assert torch.allclose(normalized_residual_degrees(degree),
                          normalized_residual_degrees(padded)[:3])
    assert normalized_residual_degrees(degree).max() == 1


def test_q_network_recomputes_residual_features_and_masks_removed():
    graph = DirectedGraph.from_edges(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    model = DirQNetwork(hidden_dim=8, message_passing_steps=2)
    q = model(graph, removed=[0])
    assert torch.isfinite(q[1:]).all()
    assert q[0] < -1e20
    q[1].backward()
    assert model.input_mlp[0].weight.grad is not None


def test_batched_q_matches_individual_graphs_and_backpropagates():
    graphs = [DirectedGraph.from_edges(4, [(0, 1), (1, 2), (2, 0), (2, 3)]),
              DirectedGraph.from_edges(3, [(0, 1), (1, 0), (1, 2)])]
    model = DirQNetwork(hidden_dim=8, message_passing_steps=2)
    removed = [(0,), ()]
    batched = model.forward_batch(graphs, removed)
    for graph, state, q in zip(graphs, removed, batched):
        expected = model(graph, state)
        assert torch.allclose(q, expected, atol=1e-5)
    (batched[0][1] + batched[1][0]).backward()
    assert model.decoder[0].weight.grad is not None


def test_n_step_return_stops_at_gscc_terminal():
    graph = DirectedGraph.from_edges(4, [(0, 1), (1, 0), (2, 3), (3, 2)])
    env = DirectedDismantlingEnv(graph)
    assert env.step(0) == pytest.approx(-0.5)
    assert not env.is_terminal()
    assert env.step(2) == pytest.approx(-0.25)
    replay = NStepReplayBuffer(8)
    replay.add_episode(env.episode, n_step=3, gamma=0.5)
    first, second = replay.transitions
    assert first.reward == pytest.approx(-0.625)
    assert first.steps == 2 and first.terminal
    assert second.reward == pytest.approx(-0.25)
    assert second.steps == 1 and second.terminal


def test_bellman_target_ignores_large_q_outside_next_gscc():
    graph = DirectedGraph.from_edges(5, [(0, 1), (1, 0), (2, 3), (3, 2), (3, 4)])
    env = DirectedDismantlingEnv(graph)
    env.step(0)
    replay = NStepReplayBuffer(4)
    replay.add_episode(env.episode, n_step=1)
    agent = DNDDrLAgent(AgentConfig(hidden_dim=8, message_passing_steps=1,
                                    learning_rate=0, batch_size=1, device="cpu"))
    prediction = float(agent.online(graph, ())[0].detach())
    agent.target.forward_batch = lambda graphs, removed: [
        torch.tensor([-5., -5., 1.5, 2.5, 100.]) for _ in graphs]
    loss = agent.optimize(replay, random.Random(1))
    assert loss == pytest.approx((prediction - (-2 / 5 + 2.5)) ** 2, rel=1e-5)


def test_train_checkpoint_and_greedy_inference(tmp_path):
    torch.manual_seed(0)
    agent = DNDDrLAgent(AgentConfig(hidden_dim=8, message_passing_steps=1,
                                    batch_size=2, device="cpu"))
    state = train(agent, TrainingConfig(episodes=2, num_min=8, num_max=9,
                                        ba_m=2, seed=4, log_interval=0))
    assert state.episodes_completed == 2
    assert agent.optimization_steps > 0
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(agent, state, path)
    restored, saved_state = load_training_checkpoint(path, device="cpu")
    assert saved_state.episodes_completed == 2
    assert restored.optimization_steps == agent.optimization_steps
    graph = directed_ba_graph(8, m=2, reciprocity=1.0, seed=3)
    solution = restored.greedy_solution(graph)
    assert len(solution) == len(set(solution))
    assert DirectedDismantlingEnv(graph).gscc_size > 1
    assert graph.gscc_size(solution) <= 1


def test_heldout_plateau_saves_best_and_stops_after_patience(tmp_path):
    agent = DNDDrLAgent(AgentConfig(hidden_dim=8, message_passing_steps=1,
                                    batch_size=2, device="cpu"))
    config = TrainingConfig(
        episodes=10, num_min=8, num_max=9, ba_m=2, seed=23,
        log_interval=0, validation_interval=2,
        validation_min_episodes=4, validation_patience=2,
        best_checkpoint=tmp_path / "best.pt",
        latest_checkpoint=tmp_path / "latest.pt",
        validation_log_file=tmp_path / "validation.csv")
    state = train(agent, config, validator=lambda model: 0.1)
    assert state.episodes_completed == 6
    assert state.best_episode == 2
    assert state.stale_validations == 2
    _, best_state = load_training_checkpoint(config.best_checkpoint, device="cpu")
    assert best_state.episodes_completed == 2
    _, latest_state = load_training_checkpoint(config.latest_checkpoint, device="cpu")
    assert latest_state.episodes_completed == 6


def test_real_high_order_graph_loads_without_losing_directions():
    graph = DirectedGraph.read_csv_edge_table(
        Path(__file__).resolve().parents[4] / "data/network-ho.csv")
    assert graph.num_nodes == 13326
    assert len(graph.edges) == 27214
    assert graph.gscc_size() == 9461
