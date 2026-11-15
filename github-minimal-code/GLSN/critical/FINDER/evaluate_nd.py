from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finder_pytorch.agent import FinderDQNAgent
from finder_pytorch.evaluate import make_progress_callback, solve_csv_edge_table, solve_edgelist
from finder_pytorch.graph import FinderGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch FINDER-ND baseline.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--edgelist", default="")
    parser.add_argument("--csv", default="")
    parser.add_argument("--source-col", default="node1")
    parser.add_argument("--target-col", default="node2")
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument(
        "--step-fraction",
        type=float,
        default=0.0,
        help="If > 0, remove this fraction of current candidate nodes each round.",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--summary-out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = FinderDQNAgent.load(args.model)

    progress_callback = None
    progress_handle = None
    log_interval = 0
    if not args.quiet and (args.log_interval > 0 or args.log_file):
        progress_callback, progress_handle = make_progress_callback(
            log_file=args.log_file or None,
            print_progress=not args.quiet,
        )
        log_interval = args.log_interval if args.log_interval > 0 else 50

    try:
        if args.csv:
            graph, solution, summary = solve_csv_edge_table(
                agent,
                args.csv,
                step_size=args.step_size,
                step_fraction=args.step_fraction,
                source_col=args.source_col,
                target_col=args.target_col,
                log_interval=log_interval,
                progress_callback=progress_callback,
            )
        elif args.edgelist:
            graph = FinderGraph.read_edgelist(args.edgelist)
            solution, robustness = solve_edgelist(
                agent,
                args.edgelist,
                step_size=args.step_size,
                step_fraction=args.step_fraction,
                log_interval=log_interval,
                progress_callback=progress_callback,
            )
            summary = {"robustness": robustness, "num_nodes": graph.num_nodes, "num_edges": len(graph.edges)}
        else:
            raise SystemExit("provide either --csv or --edgelist")
    finally:
        if progress_handle is not None:
            progress_handle.close()

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", "internal_node", "original_node"])
            for rank, node in enumerate(solution, start=1):
                writer.writerow([rank, node, graph.original_node(node)])
    if args.summary_out:
        summary_out = Path(args.summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
