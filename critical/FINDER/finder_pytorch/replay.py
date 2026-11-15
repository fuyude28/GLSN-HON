from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .env import EnvState, Episode


@dataclass(frozen=True)
class Transition:
    state: EnvState
    action: int
    reward: float
    next_state: EnvState
    terminal: bool


class NStepReplayBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.transitions: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def add_episode(self, episode: Episode, n_step: int) -> None:
        if n_step <= 0:
            raise ValueError("n_step must be positive")
        count = len(episode.actions)
        for idx in range(count):
            end = min(idx + n_step, count)
            reward = sum(episode.rewards[idx:end])
            terminal = bool(episode.terminals[end - 1]) if end else True
            next_state = episode.next_states[end - 1]
            self.add(
                Transition(
                    state=episode.states[idx],
                    action=episode.actions[idx],
                    reward=reward,
                    next_state=next_state,
                    terminal=terminal,
                )
            )

    def sample(self, batch_size: int, rng: random.Random | None = None) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.transitions) < batch_size:
            raise ValueError("not enough transitions to sample")
        chooser = rng if rng is not None else random
        return chooser.sample(list(self.transitions), batch_size)

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.add(transition)
