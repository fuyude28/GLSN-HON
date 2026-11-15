"""Offline training on small synthetic directed graphs."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from .agent import AgentConfig, DNDDrLAgent
from .graph import directed_ba_graph
from .replay import NStepReplayBuffer


@dataclass
class TrainingConfig:
    episodes: int = 1000
    num_min: int = 30
    num_max: int = 50
    ba_m: int = 2
    reciprocity: float = 0.1
    hub_source_bias: float = 0.6
    replay_capacity: int = 500_000
    n_step: int = 5
    warmup_episodes: int = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 10_000
    seed: int = 0
    log_interval: int = 100
    log_file: Path | None = None
    checkpoint_interval: int = 0
    checkpoint_dir: Path | None = None
    validation_interval: int = 0
    validation_patience: int = 5
    validation_min_episodes: int = 8000
    validation_min_delta: float = 1e-4
    validation_log_file: Path | None = None
    best_checkpoint: Path | None = None
    latest_checkpoint: Path | None = None


@dataclass
class TrainingState:
    episodes_completed: int = 0
    rng_state: tuple | None = None
    best_validation_auc: float | None = None
    best_episode: int = 0
    stale_validations: int = 0


def epsilon_by_episode(config: TrainingConfig, episode: int) -> float:
    if episode < config.warmup_episodes:
        return 1.0
    progress = min(max(episode - config.warmup_episodes, 0) /
                   max(config.epsilon_decay_episodes, 1), 1.0)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def save_training_checkpoint(
    agent: DNDDrLAgent, state: TrainingState, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": asdict(agent.config),
        "online": agent.online.state_dict(),
        "target": agent.target.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "optimization_steps": agent.optimization_steps,
        "training_state": asdict(state),
    }, path)


def load_training_checkpoint(path: Path, device: str = "auto") -> tuple[DNDDrLAgent, TrainingState]:
    agent = DNDDrLAgent.load(path, device=device)
    checkpoint = torch.load(path, map_location=agent.device, weights_only=True)
    saved = checkpoint["training_state"]
    return agent, TrainingState(
        saved["episodes_completed"], saved["rng_state"],
        saved.get("best_validation_auc"), saved.get("best_episode", 0),
        saved.get("stale_validations", 0))


def train(
    agent: DNDDrLAgent, config: TrainingConfig,
    initial_state: TrainingState | None = None,
    validator: Callable[[DNDDrLAgent], float] | None = None,
) -> TrainingState:
    rng = random.Random(config.seed)
    state = initial_state or TrainingState()
    if state.rng_state is not None:
        rng.setstate(state.rng_state)
    replay = NStepReplayBuffer(config.replay_capacity)
    columns = ["episode", "epsilon", "loss", "num_nodes", "initial_gscc",
               "removed", "replay_size", "optimization_steps"]
    log = None
    writer = None
    validation_log = None
    validation_writer = None
    if config.log_file is not None:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        log = config.log_file.open("a", encoding="utf-8", newline="")
        writer = csv.DictWriter(log, fieldnames=columns)
        if log.tell() == 0:
            writer.writeheader()
    if validator is not None and config.validation_log_file is not None:
        config.validation_log_file.parent.mkdir(parents=True, exist_ok=True)
        validation_log = config.validation_log_file.open("a", encoding="utf-8", newline="")
        validation_writer = csv.writer(validation_log)
        if validation_log.tell() == 0:
            validation_writer.writerow(("episode", "heldout_auc", "best_auc",
                                        "best_episode", "stale_validations"))
    try:
        for episode in range(state.episodes_completed, config.episodes):
            n = rng.randint(config.num_min, config.num_max)
            graph = directed_ba_graph(n, config.ba_m, config.reciprocity,
                                      config.hub_source_bias, seed=rng.randrange(2**31))
            initial_gscc = graph.gscc_size()
            epsilon = epsilon_by_episode(config, episode)
            episode_env = agent.play_episode(graph, epsilon, rng)
            replay.add_episode(episode_env.episode, config.n_step, agent.config.gamma)
            loss = agent.optimize(replay, rng) if len(replay) >= agent.config.batch_size else float("nan")
            state = TrainingState(episode + 1, rng.getstate(),
                                  state.best_validation_auc, state.best_episode,
                                  state.stale_validations)
            if episode + 1 == config.episodes or (
                config.log_interval > 0 and (episode + 1) % config.log_interval == 0
            ):
                row = dict(episode=episode + 1, epsilon=epsilon, loss=loss,
                           num_nodes=n, initial_gscc=initial_gscc,
                           removed=len(episode_env.removed), replay_size=len(replay),
                           optimization_steps=agent.optimization_steps)
                if writer is not None:
                    writer.writerow(row)
                    log.flush()
                print("episode={episode} epsilon={epsilon:.3f} loss={loss:.6g} "
                      "initial_gscc={initial_gscc} removed={removed}".format(**row), flush=True)
            if (config.checkpoint_dir is not None and config.checkpoint_interval > 0 and
                ((episode + 1) % config.checkpoint_interval == 0 or episode + 1 == config.episodes)):
                save_training_checkpoint(agent, state,
                                         config.checkpoint_dir / f"checkpoint_ep_{episode + 1:06d}.pt")
            if (validator is not None and config.validation_interval > 0 and
                (episode + 1) % config.validation_interval == 0):
                auc = validator(agent)
                improved = (state.best_validation_auc is None or
                            auc < state.best_validation_auc - config.validation_min_delta)
                if improved:
                    state.best_validation_auc = auc
                    state.best_episode = episode + 1
                    state.stale_validations = 0
                    if config.best_checkpoint is not None:
                        save_training_checkpoint(agent, state, config.best_checkpoint)
                else:
                    state.stale_validations += 1
                if config.latest_checkpoint is not None:
                    save_training_checkpoint(agent, state, config.latest_checkpoint)
                if validation_writer is not None:
                    validation_writer.writerow((episode + 1, auc,
                                                state.best_validation_auc,
                                                state.best_episode,
                                                state.stale_validations))
                    validation_log.flush()
                print(f"validation episode={episode + 1} auc={auc:.6f} "
                      f"best={state.best_validation_auc:.6f} "
                      f"best_episode={state.best_episode} "
                      f"stale={state.stale_validations}", flush=True)
                if (episode + 1 >= config.validation_min_episodes and
                    state.stale_validations >= config.validation_patience):
                    print("Stopped: held-out GSCC-AUC has plateaued", flush=True)
                    break
    finally:
        if log is not None:
            log.close()
        if validation_log is not None:
            validation_log.close()
    return state
