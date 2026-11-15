from __future__ import annotations

import argparse
from pathlib import Path

from finder_pytorch.agent import FinderDQNAgent
from finder_pytorch.training import TrainingConfig, load_training_checkpoint, train_finder_nd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch FINDER-ND baseline.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-min", type=int, default=30)
    parser.add_argument("--num-max", type=int, default=50)
    parser.add_argument("--ba-m", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-episodes", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--out", default="models/finder_nd_pytorch.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume:
        agent, initial_state = load_training_checkpoint(args.resume, device=args.device)
        print(f"resumed {args.resume} from episode {initial_state.episodes_completed}")
    else:
        agent = FinderDQNAgent(hidden_dim=args.hidden_dim, batch_size=args.batch_size, device=args.device)
        initial_state = None
    config = TrainingConfig(
        episodes=args.episodes,
        num_min=args.num_min,
        num_max=args.num_max,
        ba_m=args.ba_m,
        warmup_episodes=args.warmup_episodes,
        seed=args.seed,
        log_interval=args.log_interval,
        log_file=args.log_file or None,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir or None,
    )

    def print_progress(row: dict[str, float | int]) -> None:
        print(
            "episode={episode} epsilon={epsilon:.4f} loss={loss:.6f} "
            "replay={replay_size} opt_steps={optimization_steps} elapsed={elapsed_seconds:.1f}s".format(**row),
            flush=True,
        )

    result = train_finder_nd(agent, config, initial_state=initial_state, progress_callback=print_progress)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(out))
    print(f"saved {out} after {result.episodes_completed} episodes; final_loss={result.final_loss:.6f}")


if __name__ == "__main__":
    main()
