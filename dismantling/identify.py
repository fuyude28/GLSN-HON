"""Apply a trained DND-DRL agent to the higher-order GLSN."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from dnd_drl.agent import DNDDrLAgent
from dnd_drl.graph import DirectedGraph

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/dismantling/best.pt")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/network-ho.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/dismantling/removal_sequence.csv")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "outputs/dismantling/summary.json")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--batch-fraction", type=float, default=0.01,
                        help="Fraction of all remaining state nodes selected per Q evaluation; 0 means one-by-one.")
    parser.add_argument("--max-removals", type=int, default=0,
                        help="Optional fixed removal cap for a partial diagnostic; 0 means dismantle to termination.")
    args = parser.parse_args()

    graph = DirectedGraph.read_csv_edge_table(args.graph)
    agent = DNDDrLAgent.load(args.model, device=args.device)
    curve: list[tuple[int, int, int]] = []
    current_round = 0

    def round_progress(round_number: int, batch_size: int) -> None:
        nonlocal current_round
        current_round = round_number

    def progress(step: int, size: int) -> None:
        curve.append((step, size, current_round))
        if args.log_interval > 0 and (step == 0 or step % args.log_interval == 0 or size <= 1):
            print(f"step={step} GSCC={size} fraction={size / graph.num_nodes:.5f}", flush=True)

    start = time.perf_counter()
    sequence = agent.greedy_solution(
        graph, progress=progress, batch_fraction=args.batch_fraction,
        max_removals=args.max_removals or None, round_progress=round_progress)
    elapsed = time.perf_counter() - start
    area = sum(size for _, size, _ in curve[1:]) / (graph.num_nodes ** 2)
    complete = curve[-1][1] <= 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "batch_round", "state_node", "gscc_size_after_removal",
                         "fraction_of_initial_states_in_gscc"])
        for (rank, size, round_number), node in zip(curve[1:], sequence):
            writer.writerow([rank, round_number, graph.original_node(node), size,
                             size / graph.num_nodes])

    summary = {
        "graph": str(args.graph), "num_states": graph.num_nodes,
        "num_edges": len(graph.edges), "initial_gscc": curve[0][1],
        "batch_fraction_of_remaining_states": args.batch_fraction,
        "adaptive_rounds": current_round, "removals": len(sequence),
        "gscc_after_removals": curve[-1][1], "complete": complete,
        "auc": area if complete else None,
        "partial_area": area if not complete else None,
        "elapsed_seconds": elapsed,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
