from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch.nn import functional as F

from .env import NetworkDismantlingEnv
from .graph import FinderGraph
from .model import FinderQNetwork
from .replay import NStepReplayBuffer


@dataclass
class AgentConfig:
    hidden_dim: int = 64
    message_passing_steps: int = 3
    decoder_hidden_dim: int = 32
    learning_rate: float = 1e-4
    gamma: float = 1.0
    batch_size: int = 64
    target_update_interval: int = 1000
    device: str = "auto"


class FinderDQNAgent:
    def __init__(
        self,
        hidden_dim: int = 64,
        message_passing_steps: int = 3,
        decoder_hidden_dim: int = 32,
        learning_rate: float = 1e-4,
        gamma: float = 1.0,
        batch_size: int = 64,
        target_update_interval: int = 1000,
        device: str = "auto",
    ):
        self.config = AgentConfig(
            hidden_dim=hidden_dim,
            message_passing_steps=message_passing_steps,
            decoder_hidden_dim=decoder_hidden_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            target_update_interval=target_update_interval,
            device=str(resolve_device(device)),
        )
        self.device = resolve_device(device)
        self.online = FinderQNetwork(
            hidden_dim=hidden_dim,
            message_passing_steps=message_passing_steps,
            decoder_hidden_dim=decoder_hidden_dim,
        ).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.optimization_steps = 0

    def select_action(
        self,
        graph: FinderGraph,
        removed: Sequence[int],
        epsilon: float = 0.0,
        rng: random.Random | None = None,
    ) -> int:
        candidates = graph.candidate_nodes(removed)
        if not candidates:
            raise RuntimeError("no candidate actions")
        chooser = rng if rng is not None else random
        if chooser.random() < epsilon:
            return chooser.choice(candidates)
        with torch.no_grad():
            q = self.online.q_values(graph, removed)
        return int(max(candidates, key=lambda node: float(q[node].item())))

    def play_episode(
        self,
        graph: FinderGraph,
        epsilon: float,
        rng: random.Random | None = None,
    ) -> NetworkDismantlingEnv:
        env = NetworkDismantlingEnv(graph)
        while not env.is_terminal():
            action = self.select_action(graph, env.removed, epsilon=epsilon, rng=rng)
            env.step(action)
        return env

    def optimize(self, replay: NStepReplayBuffer) -> float:
        batch = replay.sample(self.config.batch_size)
        graphs = [transition.state.graph for transition in batch]
        removed_batch = [transition.state.removed for transition in batch]
        actions = [transition.action for transition in batch]

        pred = self.online.score_actions(graphs, removed_batch, actions)
        rewards = torch.tensor(
            [transition.reward for transition in batch], dtype=torch.float32, device=self.device
        )
        terminals = torch.tensor(
            [transition.terminal for transition in batch], dtype=torch.bool, device=self.device
        )
        next_graphs = [transition.next_state.graph for transition in batch]
        next_removed = [transition.next_state.removed for transition in batch]
        with torch.no_grad():
            next_q = self.target.max_next_q(next_graphs, next_removed)
            target = rewards + (~terminals).float() * self.config.gamma * next_q
        loss = F.mse_loss(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.optimization_steps += 1
        if self.optimization_steps % self.config.target_update_interval == 0:
            self.update_target()
        return float(loss.detach().cpu().item())

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def greedy_solution(
        self,
        graph: FinderGraph,
        step_size: int = 1,
        step_fraction: float = 0.0,
        log_interval: int = 0,
        progress_callback: Callable[[dict[str, float | int]], None] | None = None,
    ) -> list[int]:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if not 0.0 <= step_fraction <= 1.0:
            raise ValueError("step_fraction must be between 0 and 1")
        env = NetworkDismantlingEnv(graph)
        start_time = time.perf_counter()
        _emit_greedy_progress(graph, env, start_time, log_interval, progress_callback, force=True)
        while not env.is_terminal():
            with torch.no_grad():
                q = self.online.q_values(graph, env.removed)
            candidates = graph.candidate_nodes(env.removed)
            ranked = sorted(candidates, key=lambda node: float(q[node].item()), reverse=True)
            actions_this_round = _actions_this_round(len(candidates), step_size, step_fraction)
            for action in ranked[:actions_this_round]:
                if env.is_terminal():
                    break
                env.step_without_reward(action)
            _emit_greedy_progress(graph, env, start_time, log_interval, progress_callback)
        _emit_greedy_progress(graph, env, start_time, log_interval, progress_callback, force=True)
        return list(env.removed)

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config.__dict__,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimization_steps": self.optimization_steps,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "FinderDQNAgent":
        resolved_device = resolve_device(device)
        checkpoint = torch.load(path, map_location=resolved_device, weights_only=True)
        config = dict(checkpoint["config"])
        config["device"] = str(resolved_device)
        agent = cls(**config)
        agent.online.load_state_dict(checkpoint["online"])
        agent.target.load_state_dict(checkpoint["target"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.optimization_steps = int(checkpoint.get("optimization_steps", 0))
        return agent


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _actions_this_round(num_candidates: int, step_size: int, step_fraction: float) -> int:
    if step_fraction > 0.0:
        return max(1, min(num_candidates, math.ceil(num_candidates * step_fraction)))
    return min(num_candidates, step_size)


def _greedy_progress_row(graph: FinderGraph, env: NetworkDismantlingEnv, start_time: float) -> dict[str, float | int]:
    removed_count = len(env.removed)
    return {
        "step": removed_count,
        "num_nodes": graph.num_nodes,
        "lcc": graph.largest_component_size(env.removed_set),
        "residual_edges": len(graph.residual_edges(env.removed_set)),
        "elapsed_seconds": time.perf_counter() - start_time,
    }


def _emit_greedy_progress(
    graph: FinderGraph,
    env: NetworkDismantlingEnv,
    start_time: float,
    log_interval: int,
    progress_callback: Callable[[dict[str, float | int]], None] | None,
    force: bool = False,
) -> None:
    if progress_callback is None:
        return
    step = len(env.removed)
    if not force and (log_interval <= 0 or step <= 0 or step % log_interval != 0):
        return
    progress_callback(_greedy_progress_row(graph, env, start_time))
