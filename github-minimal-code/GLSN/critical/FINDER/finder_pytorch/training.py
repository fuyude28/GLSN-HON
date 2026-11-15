from __future__ import annotations

import csv
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from .agent import FinderDQNAgent
from .agent import resolve_device
from .graph import barabasi_albert_graph
from .replay import NStepReplayBuffer


@dataclass
class TrainingConfig:
    episodes: int = 1000
    replay_capacity: int = 500_000
    n_step: int = 5
    num_min: int = 30
    num_max: int = 50
    ba_m: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 10_000
    warmup_episodes: int = 10
    seed: int = 0
    log_interval: int = 100
    log_file: str | Path | None = None
    checkpoint_interval: int = 0
    checkpoint_dir: str | Path | None = None


@dataclass
class TrainingState:
    episodes_completed: int = 0
    losses: list[float] | None = None
    rng_state: object | None = None


@dataclass
class TrainingResult:
    episodes_completed: int
    start_episode: int
    losses: list[float]
    final_loss: float


def epsilon_by_episode(config: TrainingConfig, episode: int) -> float:
    if config.epsilon_decay_episodes <= 0:
        return config.epsilon_end
    frac = max(0.0, (config.epsilon_decay_episodes - episode) / config.epsilon_decay_episodes)
    return config.epsilon_end + (config.epsilon_start - config.epsilon_end) * frac


def save_training_checkpoint(
    agent: FinderDQNAgent,
    config: TrainingConfig,
    state: TrainingState,
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": {
                "config": agent.config.__dict__,
                "online": agent.online.state_dict(),
                "target": agent.target.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
                "optimization_steps": agent.optimization_steps,
            },
            "training_config": _serializable_config(config),
            "training_state": asdict(state),
        },
        out,
    )


def load_training_checkpoint(path: str | Path, device: str = "auto") -> tuple[FinderDQNAgent, TrainingState]:
    resolved_device = resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=True)
    agent_data = checkpoint["agent"]
    agent_config = dict(agent_data["config"])
    agent_config["device"] = str(resolved_device)
    agent = FinderDQNAgent(**agent_config)
    agent.online.load_state_dict(agent_data["online"])
    agent.target.load_state_dict(agent_data["target"])
    agent.optimizer.load_state_dict(agent_data["optimizer"])
    agent.optimization_steps = int(agent_data.get("optimization_steps", 0))

    state_data = checkpoint.get("training_state", {})
    return agent, TrainingState(
        episodes_completed=int(state_data.get("episodes_completed", 0)),
        losses=list(state_data.get("losses") or []),
        rng_state=_list_to_tuple(state_data.get("rng_state")),
    )


def train_finder_nd(
    agent: FinderDQNAgent,
    config: TrainingConfig,
    initial_state: TrainingState | None = None,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> TrainingResult:
    rng = random.Random(config.seed)
    replay = NStepReplayBuffer(config.replay_capacity)
    state = initial_state or TrainingState()
    if state.rng_state is not None:
        rng.setstate(state.rng_state)
    losses: list[float] = list(state.losses or [])
    start_episode = state.episodes_completed
    log_handle = _open_log_file(config.log_file)
    writer = _make_log_writer(log_handle)
    start_time = time.perf_counter()

    try:
        for episode_idx in range(start_episode, config.episodes):
            num_nodes = rng.randint(config.num_min, config.num_max)
            graph = barabasi_albert_graph(num_nodes, m=config.ba_m, seed=rng.randint(0, 2**31 - 1))
            epsilon = 1.0 if episode_idx < config.warmup_episodes else epsilon_by_episode(
                config, episode_idx - config.warmup_episodes
            )
            env = agent.play_episode(graph, epsilon=epsilon, rng=rng)
            replay.add_episode(env.episode, n_step=config.n_step)
            latest_loss = losses[-1] if losses else float("nan")
            if len(replay) >= agent.config.batch_size:
                latest_loss = agent.optimize(replay)
                losses.append(latest_loss)

            episodes_completed = episode_idx + 1
            state = TrainingState(
                episodes_completed=episodes_completed,
                losses=losses,
                rng_state=_tuple_to_list(rng.getstate()),
            )
            elapsed = time.perf_counter() - start_time
            row = {
                "episode": episodes_completed,
                "epsilon": epsilon,
                "loss": latest_loss,
                "num_nodes": num_nodes,
                "replay_size": len(replay),
                "optimization_steps": agent.optimization_steps,
                "elapsed_seconds": elapsed,
            }
            if _should_emit(episodes_completed, config.log_interval, config.episodes):
                _write_log_row(writer, row)
                if progress_callback is not None:
                    progress_callback(row)
            if _should_checkpoint(episodes_completed, config):
                save_training_checkpoint(
                    agent,
                    config,
                    state,
                    Path(config.checkpoint_dir) / f"checkpoint_ep_{episodes_completed:06d}.pt",
                )
    finally:
        if log_handle is not None:
            log_handle.close()

    return TrainingResult(
        episodes_completed=state.episodes_completed,
        start_episode=start_episode,
        losses=losses,
        final_loss=losses[-1] if losses else float("nan"),
    )


def _serializable_config(config: TrainingConfig) -> dict[str, object]:
    data = asdict(config)
    for key in ("log_file", "checkpoint_dir"):
        if data[key] is not None:
            data[key] = str(data[key])
    return data


def _tuple_to_list(value):
    if isinstance(value, tuple):
        return [_tuple_to_list(item) for item in value]
    return value


def _list_to_tuple(value):
    if isinstance(value, list):
        return tuple(_list_to_tuple(item) for item in value)
    return value


def _open_log_file(path: str | Path | None):
    if path is None:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out.open("a", encoding="utf-8", newline="")


def _make_log_writer(handle):
    if handle is None:
        return None
    fieldnames = ["episode", "epsilon", "loss", "num_nodes", "replay_size", "optimization_steps", "elapsed_seconds"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if handle.tell() == 0:
        writer.writeheader()
    return writer


def _write_log_row(writer, row: dict[str, float | int]) -> None:
    if writer is None:
        return
    writer.writerow(row)


def _should_emit(episode: int, interval: int, final_episode: int) -> bool:
    return episode == final_episode or (interval > 0 and episode % interval == 0)


def _should_checkpoint(episode: int, config: TrainingConfig) -> bool:
    return (
        config.checkpoint_dir is not None
        and config.checkpoint_interval > 0
        and (episode % config.checkpoint_interval == 0 or episode == config.episodes)
    )
