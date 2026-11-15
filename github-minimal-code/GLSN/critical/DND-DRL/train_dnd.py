"""Train DND-DRL on synthetic directed BA networks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dnd_drl.agent import AgentConfig, DNDDrLAgent
from dnd_drl.training import TrainingConfig, load_training_checkpoint, train

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-min", type=int, default=30)
    parser.add_argument("--num-max", type=int, default=50)
    parser.add_argument("--ba-m", type=int, default=2)
    parser.add_argument("--reciprocity", type=float, default=0.1)
    parser.add_argument("--hub-source-bias", type=float, default=0.6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--message-passing-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--warmup-episodes", type=int, default=10)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--log-file", type=Path, default=HERE / "logs/train.csv")
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE / "models/checkpoints")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "models/dnd_drl.pt")
    args = parser.parse_args()

    if args.resume:
        agent, state = load_training_checkpoint(args.resume, args.device)
    else:
        torch.manual_seed(args.seed)
        agent = DNDDrLAgent(AgentConfig(
            hidden_dim=args.hidden_dim,
            message_passing_steps=args.message_passing_steps,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            target_update_interval=args.target_update_interval,
            device=args.device,
        ))
        state = None
    config = TrainingConfig(
        episodes=args.episodes, num_min=args.num_min, num_max=args.num_max,
        ba_m=args.ba_m, reciprocity=args.reciprocity,
        hub_source_bias=args.hub_source_bias, n_step=args.n_step,
        warmup_episodes=args.warmup_episodes,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        seed=args.seed, log_interval=args.log_interval, log_file=args.log_file,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
    )
    result = train(agent, config, initial_state=state)
    agent.save(args.out)
    print(f"saved {args.out} after {result.episodes_completed} episodes")


if __name__ == "__main__":
    main()
