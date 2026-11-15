from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from .graph import FinderGraph


@dataclass(frozen=True)
class EnvState:
    graph: FinderGraph
    removed: tuple[int, ...]


@dataclass
class Episode:
    graph: FinderGraph
    states: list[EnvState] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    next_states: list[EnvState] = field(default_factory=list)
    terminals: list[bool] = field(default_factory=list)


class NetworkDismantlingEnv:
    """FINDER-ND environment with reward equal to negative residual LCC area."""

    def __init__(self, graph: FinderGraph):
        self.graph = graph
        self.removed: list[int] = []
        self.removed_set: set[int] = set()
        self.episode = Episode(graph)

    def state(self) -> EnvState:
        return EnvState(self.graph, tuple(self.removed))

    def candidate_actions(self) -> list[int]:
        return self.graph.candidate_nodes(self.removed_set)

    def random_action(self, rng: random.Random | None = None) -> int:
        candidates = self.candidate_actions()
        if not candidates:
            raise RuntimeError("no available actions in terminal environment")
        chooser = rng if rng is not None else random
        return chooser.choice(candidates)

    def is_terminal(self) -> bool:
        return not self.graph.has_residual_edges(self.removed_set)

    def step(self, action: int) -> float:
        if action in self.removed_set:
            raise ValueError(f"node {action} has already been removed")
        if action not in self.candidate_actions():
            raise ValueError(f"node {action} is not a useful residual action")

        state = self.state()
        self.removed.append(action)
        self.removed_set.add(action)
        reward = self.current_reward()
        next_state = self.state()

        self.episode.states.append(state)
        self.episode.actions.append(action)
        self.episode.rewards.append(reward)
        self.episode.next_states.append(next_state)
        self.episode.terminals.append(self.is_terminal())
        return reward

    def step_without_reward(self, action: int) -> None:
        if action in self.removed_set:
            return
        self.removed.append(action)
        self.removed_set.add(action)

    def current_reward(self) -> float:
        return -self.graph.largest_component_size(self.removed_set) / (
            self.graph.num_nodes * self.graph.num_nodes
        )

    @staticmethod
    def complete_solution(graph: FinderGraph, partial_solution: Sequence[int]) -> list[int]:
        selected = list(dict.fromkeys(int(node) for node in partial_solution))
        selected_set = set(selected)
        selected.extend(node for node in range(graph.num_nodes) if node not in selected_set)
        return selected

    @staticmethod
    def robustness(graph: FinderGraph, solution: Sequence[int]) -> float:
        removed: set[int] = set()
        total = 0.0
        for node in solution:
            removed.add(int(node))
            total += graph.largest_component_size(removed)
        return total / (graph.num_nodes * graph.num_nodes)
