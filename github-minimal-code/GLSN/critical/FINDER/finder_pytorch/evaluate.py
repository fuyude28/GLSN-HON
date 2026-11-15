from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from .agent import FinderDQNAgent
from .env import NetworkDismantlingEnv
from .graph import FinderGraph


def evaluate_solution(graph: FinderGraph, partial_solution: list[int]) -> float:
    solution = NetworkDismantlingEnv.complete_solution(graph, partial_solution)
    return NetworkDismantlingEnv.robustness(graph, solution)


def summarize_solution(graph: FinderGraph, partial_solution: list[int]) -> dict[str, float | int]:
    complete = NetworkDismantlingEnv.complete_solution(graph, partial_solution)
    return {
        "num_nodes": graph.num_nodes,
        "num_edges": len(graph.edges),
        "initial_lcc": graph.largest_component_size([]),
        "removed_until_terminal": len(partial_solution),
        "removed_fraction_until_terminal": len(partial_solution) / max(graph.num_nodes, 1),
        "robustness": NetworkDismantlingEnv.robustness(graph, complete),
    }


def solve_edgelist(
    agent: FinderDQNAgent,
    path: str | Path,
    step_size: int = 1,
    step_fraction: float = 0.0,
    log_interval: int = 0,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> tuple[list[int], float]:
    graph = FinderGraph.read_edgelist(path)
    solution = agent.greedy_solution(
        graph,
        step_size=step_size,
        step_fraction=step_fraction,
        log_interval=log_interval,
        progress_callback=progress_callback,
    )
    return solution, evaluate_solution(graph, solution)


def solve_csv_edge_table(
    agent: FinderDQNAgent,
    path: str | Path,
    step_size: int = 1,
    step_fraction: float = 0.0,
    source_col: str = "node1",
    target_col: str = "node2",
    log_interval: int = 0,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> tuple[FinderGraph, list[int], dict[str, float | int]]:
    graph = FinderGraph.read_csv_edge_table(path, source_col=source_col, target_col=target_col)
    solution = agent.greedy_solution(
        graph,
        step_size=step_size,
        step_fraction=step_fraction,
        log_interval=log_interval,
        progress_callback=progress_callback,
    )
    return graph, solution, summarize_solution(graph, solution)


def make_progress_callback(
    log_file: str | Path | None = None,
    print_progress: bool = True,
) -> tuple[Callable[[dict[str, float | int]], None], object | None]:
    handle = _open_progress_log(log_file)
    writer = _make_progress_writer(handle)

    def callback(row: dict[str, float | int]) -> None:
        if print_progress:
            print(
                "step={step} lcc={lcc} residual_edges={residual_edges} elapsed={elapsed_seconds:.1f}s".format(
                    **row
                ),
                flush=True,
            )
        if writer is not None:
            writer.writerow(row)
            if handle is not None:
                handle.flush()

    return callback, handle


def _open_progress_log(path: str | Path | None):
    if path is None:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out.open("a", encoding="utf-8", newline="")


def _make_progress_writer(handle):
    if handle is None:
        return None
    fieldnames = ["step", "num_nodes", "lcc", "residual_edges", "elapsed_seconds"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if handle.tell() == 0:
        writer.writeheader()
    return writer
