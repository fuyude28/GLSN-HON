"""Train DND-DRL with the manuscript's validation and early-stopping protocol."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import torch

from dnd_drl.agent import AgentConfig, DNDDrLAgent
from dnd_drl.graph import directed_ba_graph
from dnd_drl.training import TrainingConfig, train

ROOT = Path(__file__).resolve().parents[1]


def heldout_graphs(count: int, seed: int, n_min: int, n_max: int,
                   ba_m: int, reciprocity: float, hub_bias: float) -> list:
    rng = random.Random(seed)
    return [directed_ba_graph(rng.randint(n_min, n_max), ba_m, reciprocity,
                              hub_bias, seed=rng.randrange(2**31))
            for _ in range(count)]


def areas(agent: DNDDrLAgent, graphs: list) -> list[float]:
    values = []
    for graph in graphs:
        curve = []
        agent.greedy_solution(graph, batch_fraction=0.01,
                              progress=lambda step, size: curve.append(size))
        values.append(sum(curve[1:]) / graph.num_nodes ** 2)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-episodes", type=int, default=30000)
    parser.add_argument("--num-min", type=int, default=30)
    parser.add_argument("--num-max", type=int, default=100)
    parser.add_argument("--ba-m", type=int, default=2)
    parser.add_argument("--reciprocity", type=float, default=0.1)
    parser.add_argument("--hub-source-bias", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=30000)
    parser.add_argument("--test-seed", type=int, default=40000)
    parser.add_argument("--validation-graphs", type=int, default=48)
    parser.add_argument("--test-graphs", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=1000)
    parser.add_argument("--min-episodes", type=int, default=8000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.00015)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/dismantling")
    args = parser.parse_args()
    if args.validation_interval < 1 or args.max_episodes < args.validation_interval:
        parser.error("max-episodes must allow at least one validation interval")
    if args.validation_graphs < 1 or args.test_graphs < 1:
        parser.error("validation-graphs and test-graphs must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    validation = heldout_graphs(args.validation_graphs, args.validation_seed,
                                args.num_min, args.num_max, args.ba_m,
                                args.reciprocity, args.hub_source_bias)
    heldout_test = heldout_graphs(args.test_graphs, args.test_seed,
                                  args.num_min, args.num_max, args.ba_m,
                                  args.reciprocity, args.hub_source_bias)
    agent = DNDDrLAgent(AgentConfig(device=args.device, batch_size=args.batch_size))
    config = TrainingConfig(
        episodes=args.max_episodes, num_min=args.num_min, num_max=args.num_max,
        ba_m=args.ba_m, reciprocity=args.reciprocity,
        hub_source_bias=args.hub_source_bias, seed=args.seed,
        epsilon_decay_episodes=5000, warmup_episodes=50,
        log_interval=args.validation_interval, log_file=args.out / "train.csv",
        validation_interval=args.validation_interval,
        validation_patience=args.patience,
        validation_min_episodes=args.min_episodes,
        validation_min_delta=args.min_delta,
        validation_log_file=args.out / "validation.csv",
        best_checkpoint=args.out / "best.pt", latest_checkpoint=args.out / "latest.pt")
    state = train(agent, config, validator=lambda candidate: mean(areas(candidate, validation)))
    best_agent = DNDDrLAgent.load(config.best_checkpoint, device=args.device)
    test_areas = areas(best_agent, heldout_test)
    with (args.out / "test.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["graph", "nodes", "dnd_drl_auc"])
        writer.writerows((i, graph.num_nodes, auc)
                         for i, (graph, auc) in enumerate(zip(heldout_test, test_areas)))
    report = dict(method="DND-DRL", device=str(agent.device),
                  arguments=vars(args), agent_config=asdict(agent.config),
                  training_config=asdict(config),
                  episodes_completed=state.episodes_completed,
                  best_episode=state.best_episode, best_validation_auc=state.best_validation_auc,
                  stale_validations=state.stale_validations,
                  test_mean_auc=mean(test_areas))
    (args.out / "training.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
