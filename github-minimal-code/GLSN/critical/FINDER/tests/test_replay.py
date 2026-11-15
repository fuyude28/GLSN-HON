from finder_pytorch.env import NetworkDismantlingEnv
from finder_pytorch.graph import FinderGraph
from finder_pytorch.replay import NStepReplayBuffer


def test_replay_adds_n_step_transitions_from_finished_episode():
    graph = FinderGraph.from_edges(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])
    env = NetworkDismantlingEnv(graph)
    env.step(1)
    env.step(2)

    replay = NStepReplayBuffer(capacity=10)
    replay.add_episode(env.episode, n_step=2)

    assert len(replay) == 2
    first = replay.transitions[0]
    second = replay.transitions[1]
    assert first.state.removed == ()
    assert first.action == 1
    assert first.next_state.removed == (1, 2)
    assert first.terminal is True
    assert first.reward == env.episode.rewards[0] + env.episode.rewards[1]
    assert second.state.removed == (1,)
    assert second.action == 2
    assert second.terminal is True
