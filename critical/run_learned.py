"""Evaluate available pretrained DND and FINDER on directed GSCC removal."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE / "DND"), str(HERE.parent / "FINDER"),
                str(HERE.parent / "DND-DRL")]

from dnd_drl.env import DirectedDismantlingEnv
from dnd_drl.graph import DirectedGraph
from finder_pytorch.agent import FinderDQNAgent
from finder_pytorch.graph import FinderGraph
from run_structural import evaluate
from tad_pretrained import score_pretrained


def finder_curve(graph: DirectedGraph, checkpoint: Path, curve_path: Path,
                 batch_fraction: float) -> dict:
    agent = FinderDQNAgent.load(str(checkpoint), device="cpu")
    agent.online.eval()
    projection = FinderGraph.from_edges(graph.num_nodes, graph.edges)
    env = DirectedDismantlingEnv(graph)
    start = time.perf_counter()
    area = 0.0
    rounds = 0
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("batch_round", "removed", "removed_state", "gscc_size",
                         "gscc_fraction", "cumulative_auc"))
        writer.writerow((0, 0, "", env.gscc_size,
                         env.gscc_size / graph.num_nodes, 0))
        while not env.is_terminal():
            with torch.no_grad():
                q = agent.online.q_values(projection, env.removed_set,
                                          mask_removed=False).tolist()
            candidates = env.candidate_actions()
            count = min(len(candidates), max(1, math.ceil(
                batch_fraction * (graph.num_nodes - len(env.removed)))))
            selected = sorted(candidates, key=lambda node: (-q[node], node))[:count]
            rounds += 1
            for step, size in env.remove_batch(selected):
                area += size / graph.num_nodes ** 2
                writer.writerow((rounds, step, graph.original_node(env.removed[step - 1]),
                                 size, size / graph.num_nodes, area))
            if rounds % 10 == 0:
                print(f"FINDER round={rounds} removed={len(env.removed)} "
                      f"GSCC={env.gscc_size}", flush=True)
    return {"num_states": graph.num_nodes, "num_edges": len(graph.edges),
            "initial_gscc": graph.gscc_size(), "batch_fraction": batch_fraction,
            "adaptive_rounds": rounds, "removals": len(env.removed),
            "auc": area, "evaluation_seconds": time.perf_counter() - start,
            "projection": "unweighted undirected, original FINDER-ND input",
            "ranking": "dynamic Q scores on residual graph, GSCC candidates"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dnd", "finder"), required=True)
    parser.add_argument("--graph", type=Path,
                        default=ROOT / "data/network-ho.csv")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-fraction", type=float, default=0.01)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    graph = DirectedGraph.read_csv_edge_table(args.graph)
    output = (HERE / "DND" if args.method == "dnd"
              else HERE.parent / "FINDER") / "results"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or (HERE / "TAD" / "Code" / "model_checkpoint_DND.pth"
        if args.method == "dnd" else HERE.parent / "FINDER" / "models" / "finder_nd_ba_20k.pt")
    if args.method == "dnd":
        start = time.perf_counter()
        scores = score_pretrained(graph, checkpoint)
        with (output / "dnd_scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("state_node", "score"))
            writer.writerows((graph.original_node(node), float(score))
                             for node, score in enumerate(scores))
        score_seconds = time.perf_counter() - start
        print(f"DND scored {len(scores)} states in {score_seconds:.1f}s",
              flush=True)
        order = sorted(range(graph.num_nodes),
                       key=lambda node: (-float(scores[node]), node))
        summary = evaluate(graph, order, output / "dnd_curve.csv", args.batch_fraction)
        summary.update(ranking="static DND probabilities, GSCC candidates",
                       score_seconds=score_seconds,
                       provenance="DND comparison checkpoint distributed with TAD; "
                       "four raw features, not the locally untrained SPL DND implementation")
    else:
        summary = finder_curve(graph, checkpoint, output / "finder_curve.csv",
                               args.batch_fraction)
        summary["provenance"] = "local FINDER-ND PyTorch rewrite, BA 20k episodes"
    summary.update(method=args.method, checkpoint=str(checkpoint))
    (output / f"{args.method}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
