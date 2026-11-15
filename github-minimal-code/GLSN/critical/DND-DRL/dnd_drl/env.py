"""MDP with state-node removal and normalized GSCC penalty."""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import DirectedGraph


@dataclass(frozen=True)
class EnvState:
    graph: DirectedGraph
    removed: tuple[int, ...]


@dataclass
class Episode:
    states: list[EnvState] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    next_states: list[EnvState] = field(default_factory=list)
    terminals: list[bool] = field(default_factory=list)


class DirectedDismantlingEnv:
    def __init__(self, graph: DirectedGraph):
        self.graph = graph
        self.removed: list[int] = []
        self.removed_set: set[int] = set()
        self.gscc_size, self._candidates = graph.gscc_state()
        self.episode = Episode()

    def state(self) -> EnvState:
        return EnvState(self.graph, tuple(self.removed))

    def candidate_actions(self) -> list[int]:
        return list(self._candidates)

    def is_terminal(self) -> bool:
        # No nontrivial strongly connected component remains.
        return self.gscc_size <= 1

    def remove(self, action: int, record: bool = True) -> float:
        if self.is_terminal():
            raise RuntimeError("GSCC has already been eliminated")
        if action not in self._candidates:
            raise ValueError(f"node {action} is not in a current largest SCC")
        before = self.state() if record else None
        self.removed.append(action)
        self.removed_set.add(action)
        self.gscc_size, self._candidates = self.graph.gscc_state(self.removed_set)
        reward = -self.gscc_size / self.graph.num_nodes
        if record:
            self.episode.states.append(before)
            self.episode.actions.append(action)
            self.episode.rewards.append(reward)
            self.episode.next_states.append(self.state())
            self.episode.terminals.append(self.is_terminal())
        return reward

    def step(self, action: int) -> float:
        return self.remove(action)

    def remove_batch(self, actions: list[int]) -> list[tuple[int, int]]:
        """Apply one Q-ranked batch without reselecting its candidates."""
        selected = set(self._candidates)
        if len(set(actions)) != len(actions) or any(node not in selected for node in actions):
            raise ValueError("batch must contain distinct current largest-SCC candidates")
        progress = []
        for node in actions:
            if self.is_terminal():
                break
            self.removed.append(node)
            self.removed_set.add(node)
            self.gscc_size, self._candidates = self.graph.gscc_state(self.removed_set)
            progress.append((len(self.removed), self.gscc_size))
        return progress
