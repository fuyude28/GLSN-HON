"""Epsilon-greedy DQN agent and greedy dismantling policy."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.nn import functional as F

from .env import DirectedDismantlingEnv
from .graph import DirectedGraph
from .model import DirQNetwork
from .replay import NStepReplayBuffer


@dataclass
class AgentConfig:
    hidden_dim: int = 64
    message_passing_steps: int = 3
    learning_rate: float = 1e-4
    gamma: float = 1.0
    batch_size: int = 64
    target_update_interval: int = 1000
    device: str = "auto"


class DNDDrLAgent:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        selected = ("cuda" if torch.cuda.is_available() else "cpu") if self.config.device == "auto" else self.config.device
        self.device = torch.device(selected)
        self.online = DirQNetwork(self.config.hidden_dim, self.config.message_passing_steps).to(self.device)
        self.target = copy.deepcopy(self.online)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.optimization_steps = 0

    def select_action(
        self, graph: DirectedGraph, removed: list[int], epsilon: float,
        rng: random.Random, candidates: list[int] | None = None
    ) -> int:
        candidates = candidates if candidates is not None else graph.gscc_candidates(removed)
        if not candidates:
            raise RuntimeError("no nontrivial GSCC candidates remain")
        if rng.random() < epsilon:
            return rng.choice(candidates)
        with torch.no_grad():
            q = self.online(graph, removed)
        return candidates[int(q[candidates].argmax().item())]

    def play_episode(self, graph: DirectedGraph, epsilon: float, rng: random.Random) -> DirectedDismantlingEnv:
        env = DirectedDismantlingEnv(graph)
        while not env.is_terminal():
            env.step(self.select_action(graph, env.removed, epsilon, rng,
                                        env.candidate_actions()))
        return env

    def optimize(self, replay: NStepReplayBuffer, rng: random.Random) -> float:
        batch = replay.sample(self.config.batch_size, rng)
        current = self.online.forward_batch(
            [t.state.graph for t in batch], [t.state.removed for t in batch])
        predicted = torch.stack([
            values[t.action] for values, t in zip(current, batch)])
        with torch.no_grad():
            pending = [t for t in batch if not t.terminal]
            next_scores = self.target.forward_batch(
                [t.next_state.graph for t in pending],
                [t.next_state.removed for t in pending]) if pending else []
            next_iter = iter(next_scores)
            next_values = torch.stack([
                predicted.new_zeros(()) if t.terminal else
                next(next_iter)[t.next_state.graph.gscc_candidates(t.next_state.removed)].max()
                for t in batch])
            rewards = predicted.new_tensor([t.reward for t in batch])
            discounts = predicted.new_tensor([
                0.0 if t.terminal else self.config.gamma ** t.steps for t in batch
            ])
            targets = rewards + discounts * next_values
        loss = F.mse_loss(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.optimization_steps += 1
        if self.optimization_steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().item())

    def greedy_solution(
        self, graph: DirectedGraph,
        progress: Callable[[int, int], None] | None = None,
        batch_fraction: float = 0.01,
        max_removals: int | None = None,
        round_progress: Callable[[int, int], None] | None = None,
    ) -> list[int]:
        if not 0 <= batch_fraction <= 1:
            raise ValueError("batch_fraction must be between 0 and 1")
        env = DirectedDismantlingEnv(graph)
        if progress is not None:
            progress(0, env.gscc_size)
        rounds = 0
        while not env.is_terminal() and (max_removals is None or len(env.removed) < max_removals):
            remaining = graph.num_nodes - len(env.removed)
            candidates = env.candidate_actions()
            batch_size = min(len(candidates), max(1, math.ceil(batch_fraction * remaining)))
            if max_removals is not None:
                batch_size = min(batch_size, max_removals - len(env.removed))
            with torch.no_grad():
                q = self.online(graph, env.removed)
            ranked = torch.argsort(q[candidates], descending=True, stable=True)[:batch_size]
            selected = [candidates[index] for index in ranked.tolist()]
            applied = env.remove_batch(selected)
            rounds += 1
            if round_progress is not None:
                round_progress(rounds, len(applied))
            if progress is not None:
                for step, size in applied:
                    progress(step, size)
        return env.removed

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "online": self.online.state_dict(),
                    "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "optimization_steps": self.optimization_steps}, out)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> DNDDrLAgent:
        selected = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        checkpoint = torch.load(path, map_location=selected, weights_only=True)
        config = AgentConfig(**checkpoint["config"])
        config.device = selected
        agent = cls(config)
        agent.online.load_state_dict(checkpoint["online"])
        agent.target.load_state_dict(checkpoint["target"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.optimization_steps = checkpoint["optimization_steps"]
        return agent
