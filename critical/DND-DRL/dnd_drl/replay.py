"""N-step replay adapted from the FINDER training loop."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .env import EnvState, Episode


@dataclass(frozen=True)
class Transition:
    state: EnvState
    action: int
    reward: float
    next_state: EnvState
    terminal: bool
    steps: int


class NStepReplayBuffer:
    def __init__(self, capacity: int):
        self.transitions: list[Transition] = []
        self.capacity = capacity
        self.next_index = 0

    def __len__(self) -> int:
        return len(self.transitions)

    def add_episode(self, episode: Episode, n_step: int, gamma: float = 1.0) -> None:
        if n_step < 1:
            raise ValueError("n_step must be positive")
        for start in range(len(episode.actions)):
            end = min(start + n_step, len(episode.actions))
            reward = sum(gamma ** offset * r for offset, r in
                         enumerate(episode.rewards[start:end]))
            transition = Transition(
                state=episode.states[start], action=episode.actions[start],
                reward=reward, next_state=episode.next_states[end - 1],
                terminal=episode.terminals[end - 1], steps=end - start,
            )
            if len(self.transitions) < self.capacity:
                self.transitions.append(transition)
            else:
                self.transitions[self.next_index] = transition
            self.next_index = (self.next_index + 1) % self.capacity

    def sample(self, batch_size: int, rng: random.Random) -> list[Transition]:
        return [self.transitions[index]
                for index in rng.sample(range(len(self.transitions)), batch_size)]
