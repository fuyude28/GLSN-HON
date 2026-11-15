"""CoreHD dynamic dismantling for the ship network (pure-Python rewrite).

CoreHD (Zdeborova, Zhang, Zhou, Sci. Rep. 2016) is an intrinsically iterative
dismantling method:

1. Decycling on the 2-core: repeatedly delete the node with the highest degree in
   the current 2-core, then re-prune all degree<=1 nodes (recompute the 2-core).
   This continues until the 2-core is empty, i.e. the remaining graph is a forest.
2. Tree-breaking: break every remaining tree component whose size exceeds
   ``csize`` by deleting the centroid-like node that minimises the largest
   resulting sub-tree, recursively, until all components are <= ``csize``.

This rewrite follows the reference C++ implementation ``TAbyTwoCoreV01.cpp`` for
phases 1 and 2 but deliberately omits the greedy *reinsertion* post-processing.

To stay comparable with the FINDER_pytorch / HDC / HBC baselines (which dismantle
until no residual edges remain), ``csize`` defaults to 1, which fully fragments the
forest so that residual edges reach 0. Set ``--csize`` to e.g. ceil(0.01 * N) for
the paper's component-size-threshold variant.

Robustness definitions are kept consistent with ``FINDER_pytorch``,
``scripts/weighted_robustness.py`` and the HDC/HBC baselines:
- connectivity uses the undirected largest connected component (LCC);
- standard ND robustness = sum_t LCC(t) / N^2 over the full removal order;
- weighted (cost) robustness = area under the (cost, survival) curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable


EdgeKey = tuple[str, str]
Adjacency = dict[str, dict[str, float]]


def read_directed_edges(
    path: str | Path,
    source_col: str,
    target_col: str,
    weight_col: str,
) -> tuple[list[tuple[str, str, float]], set[str]]:
    """Read the raw directed edge table (node1 -> node2) with weights."""
    edges: list[tuple[str, str, float]] = []
    nodes: set[str] = set()
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {source_col, target_col, weight_col}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        for row in reader:
            u = row[source_col].strip()
            v = row[target_col].strip()
            if not u or not v or u == v:
                continue
            weight = float(row[weight_col])
            edges.append((u, v, weight))
            nodes.add(u)
            nodes.add(v)
    return edges, nodes


def build_undirected_weighted(
    edges: Iterable[tuple[str, str, float]],
) -> dict[EdgeKey, float]:
    """Collapse directed edges into undirected weighted edges (summing weights)."""
    edge_weights: dict[EdgeKey, float] = defaultdict(float)
    for u, v, weight in edges:
        key = (u, v) if u < v else (v, u)
        edge_weights[key] += weight
    return dict(edge_weights)


def normalize_weights(edge_weights: dict[EdgeKey, float], method: str) -> dict[EdgeKey, float]:
    if not edge_weights:
        return {}
    if method == "none":
        return dict(edge_weights)
    if method == "sum":
        total = sum(edge_weights.values())
        if total <= 0:
            raise ValueError("edge weight sum must be positive")
        return {edge: weight / total for edge, weight in edge_weights.items()}
    if method == "minmax":
        values = list(edge_weights.values())
        lo = min(values)
        hi = max(values)
        if hi == lo:
            return {edge: 1.0 for edge in edge_weights}
        return {edge: (weight - lo) / (hi - lo) for edge, weight in edge_weights.items()}
    raise ValueError(f"unknown normalization method: {method}")


def build_adjacency(edge_weights: dict[EdgeKey, float]) -> Adjacency:
    adjacency: Adjacency = defaultdict(dict)
    for (u, v), weight in edge_weights.items():
        if weight <= 0:
            continue
        adjacency[u][v] = adjacency[u].get(v, 0.0) + weight
        adjacency[v][u] = adjacency[v].get(u, 0.0) + weight
    return {node: dict(neighbors) for node, neighbors in adjacency.items()}


def largest_component_size(active: set[str], adjacency: Adjacency) -> int:
    if not active:
        return 0
    seen: set[str] = set()
    best = 0
    for start in active:
        if start in seen:
            continue
        size = 0
        queue = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in adjacency.get(node, {}):
                if neighbor in active and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        best = max(best, size)
    return best


def residual_edge_count(active: set[str], adjacency: Adjacency) -> int:
    return sum(1 for u in active for v in adjacency.get(u, {}) if v in active and u < v)


def has_residual_edges(active: set[str], adjacency: Adjacency) -> bool:
    for u in active:
        for v in adjacency.get(u, {}):
            if v in active:
                return True
    return False


def two_core(active: set[str], adjacency: Adjacency) -> tuple[set[str], dict[str, int]]:
    """Return the 2-core of the subgraph induced by ``active`` and its core degrees.

    Iteratively removes degree<=1 nodes. The returned degree dict gives, for every
    node remaining in the core, its degree counted within the core.
    """
    degree: dict[str, int] = {n: sum(1 for nb in adjacency[n] if nb in active) for n in active}
    core = set(active)
    stack = [n for n in core if degree[n] <= 1]
    while stack:
        u = stack.pop()
        if u in core and degree[u] <= 1:
            core.discard(u)
            for nb in adjacency[u]:
                if nb in core:
                    degree[nb] -= 1
                    if degree[nb] <= 1:
                        stack.append(nb)
    return core, {n: degree[n] for n in core}


def decycle(adjacency: Adjacency, step_fraction: float = 0.1) -> list[str]:
    """CoreHD decycling phase (batched, 10% of the 2-core per round).

    Each round, compute the current 2-core and delete its top ``step_fraction``
    fraction of nodes (ranked by core degree, ties broken by smallest id),
    then recompute the 2-core. Continues until the 2-core is empty. Pruned leaves
    are kept (not deleted). Returns the ordered list of deleted nodes.
    """
    if not 0.0 < step_fraction <= 1.0:
        raise ValueError("step_fraction must be in (0, 1]")
    active: set[str] = set(adjacency)
    removed: list[str] = []
    while True:
        core, degree = two_core(active, adjacency)
        if not core:
            break
        k = max(1, min(len(core), math.ceil(len(core) * step_fraction)))
        batch = sorted(core, key=lambda n: (-degree[n], n))[:k]
        for node in batch:
            active.discard(node)
            removed.append(node)
    return removed


def _components(nodes: Iterable[str], adjacency: Adjacency, allowed: set[str]) -> list[list[str]]:
    seen: set[str] = set()
    comps: list[list[str]] = []
    for start in nodes:
        if start in seen or start not in allowed:
            continue
        comp: list[str] = []
        stack = [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for nb in adjacency[u]:
                if nb in allowed and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def _tree_centroid(comp: list[str], adjacency: Adjacency, allowed: set[str]) -> str:
    """Return the node whose removal minimises the largest resulting sub-tree."""
    comp_set = set(comp)
    n = len(comp)
    root = comp[0]
    parent: dict[str, str | None] = {root: None}
    order: list[str] = []
    stack = [root]
    seen = {root}
    while stack:
        u = stack.pop()
        order.append(u)
        for nb in adjacency[u]:
            if nb in comp_set and nb in allowed and nb not in seen:
                seen.add(nb)
                parent[nb] = u
                stack.append(nb)

    size: dict[str, int] = {}
    for u in reversed(order):
        total = 1
        for nb in adjacency[u]:
            if nb in comp_set and nb in allowed and parent.get(nb) == u:
                total += size[nb]
        size[u] = total

    best: str | None = None
    best_val: int | None = None
    for u in order:
        max_sub = n - size[u]
        for nb in adjacency[u]:
            if nb in comp_set and nb in allowed and parent.get(nb) == u:
                if size[nb] > max_sub:
                    max_sub = size[nb]
        if best_val is None or max_sub < best_val or (max_sub == best_val and u < best):
            best_val = max_sub
            best = u
    assert best is not None
    return best


def tree_break(
    active: set[str],
    adjacency: Adjacency,
    csize: int,
    removed: list[str],
) -> None:
    """CoreHD tree-breaking phase (no reinsertion).

    Operates in place on the forest induced by ``active``: deletes centroid nodes
    until every connected component has at most ``csize`` nodes. Deleted nodes are
    appended to ``removed``.
    """
    if csize < 1:
        raise ValueError("csize must be >= 1")
    worklist = [comp for comp in _components(list(active), adjacency, active) if len(comp) > csize]
    while worklist:
        comp = [u for u in worklist.pop() if u in active]
        if len(comp) <= csize:
            continue
        centroid = _tree_centroid(comp, adjacency, active)
        active.discard(centroid)
        removed.append(centroid)
        for sub in _components(comp, adjacency, active):
            if len(sub) > csize:
                worklist.append(sub)


def corehd_dismantle_order(
    adjacency: Adjacency,
    step_fraction: float = 0.1,
    csize: int = 1,
) -> list[str]:
    """Full CoreHD dismantling order: batched 10% decycling, then tree-breaking."""
    removed = decycle(adjacency, step_fraction)
    active = set(adjacency) - set(removed)
    tree_break(active, adjacency, csize, removed)
    return removed


def complete_order(order: list[str], all_nodes: Iterable[str]) -> list[str]:
    seen = set(order)
    complete = list(order)
    for node in all_nodes:
        if node not in seen:
            complete.append(node)
            seen.add(node)
    return complete


def standard_robustness(adjacency: Adjacency, order: list[str], num_nodes: int) -> float:
    """ND robustness = sum_t LCC(t) / N^2 over the full removal order."""
    active = set(adjacency)
    total = 0.0
    for node in order:
        if node in active:
            active.remove(node)
        total += largest_component_size(active, adjacency)
    return total / (num_nodes * num_nodes)


def curve_area(points: Iterable[dict[str, float | int]]) -> float:
    rows = list(points)
    area = 0.0
    for prev, cur in zip(rows, rows[1:]):
        x0 = float(prev["cost"])
        x1 = float(cur["cost"])
        y0 = float(prev["survival"])
        y1 = float(cur["survival"])
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def remove_node_cost(node: str, active: set[str], adjacency: Adjacency) -> float:
    if node not in active:
        return 0.0
    return sum(weight for neighbor, weight in adjacency.get(node, {}).items() if neighbor in active)


def compute_weighted_curve(
    edge_weights: dict[EdgeKey, float],
    order: list[str],
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    adjacency = build_adjacency(edge_weights)
    active = set(adjacency)
    total_cost = sum(edge_weights.values())
    if total_cost <= 0:
        raise ValueError("total normalized edge cost must be positive")
    initial_lcc = largest_component_size(active, adjacency)
    rows: list[dict[str, float | int]] = [
        {
            "rank": 0,
            "node": "",
            "step_cost": 0.0,
            "cost": 0.0,
            "lcc": initial_lcc,
            "survival": 1.0,
            "residual_edges": residual_edge_count(active, adjacency),
        }
    ]

    cumulative_cost = 0.0
    removed_count = 0
    for node in order:
        if node not in active:
            continue
        removed_count += 1
        step_cost = remove_node_cost(node, active, adjacency)
        cumulative_cost += step_cost
        active.remove(node)
        lcc = largest_component_size(active, adjacency)
        rows.append(
            {
                "rank": removed_count,
                "node": node,
                "step_cost": step_cost,
                "cost": min(cumulative_cost / total_cost, 1.0),
                "lcc": lcc,
                "survival": lcc / max(initial_lcc, 1),
                "residual_edges": residual_edge_count(active, adjacency),
            }
        )
        if rows[-1]["residual_edges"] == 0:
            break

    summary = {
        "num_nodes": len(adjacency),
        "num_edges": len(edge_weights),
        "total_normalized_edge_weight": total_cost,
        "initial_lcc": initial_lcc,
        "removed_until_terminal": removed_count,
        "removed_fraction_until_terminal": removed_count / max(len(adjacency), 1),
        "final_cost": rows[-1]["cost"],
        "final_lcc": rows[-1]["lcc"],
        "final_residual_edges": rows[-1]["residual_edges"],
        "weighted_robustness": curve_area(rows),
    }
    return rows, summary


def write_order(path: str | Path, order: list[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "original_node"])
        for rank, node in enumerate(order, start=1):
            writer.writerow([rank, node])


def write_curve(path: str | Path, rows: list[dict[str, float | int]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "node", "step_cost", "cost", "lcc", "survival", "residual_edges"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_plot(
    path: str | Path,
    rows: list[dict[str, float | int]],
    robustness: float,
    title: str = "CoreHD Weighted Robustness",
) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    width = 720
    height = 520
    left = 80
    right = 30
    top = 35
    bottom = 75
    plot_width = width - left - right
    plot_height = height - top - bottom

    def project(cost: float, survival: float) -> tuple[float, float]:
        x = left + max(0.0, min(cost, 1.0)) * plot_width
        y = top + (1.0 - max(0.0, min(survival, 1.0))) * plot_height
        return x, y

    points = [project(float(row["cost"]), float(row["survival"])) for row in rows]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    grid_lines = []
    tick_labels = []
    for idx in range(6):
        value = idx / 5
        x = left + value * plot_width
        y = top + (1.0 - value) * plot_height
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#ddd" />'
        )
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ddd" />'
        )
        tick_labels.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-size="13">{value:.1f}</text>'
        )
        tick_labels.append(
            f'<text x="{left - 14}" y="{y + 5:.2f}" text-anchor="end" font-size="13">{value:.1f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-weight="bold">{title}</text>
  {"".join(grid_lines)}
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>
  {"".join(tick_labels)}
  <polyline points="{polyline}" fill="none" stroke="#9467bd" stroke-width="2.4"/>
  <text x="{left + plot_width - 8}" y="{top + 22}" text-anchor="end" font-size="14">R = {robustness:.6f}</text>
  <text x="{left + plot_width / 2}" y="{height - 22}" text-anchor="middle" font-size="15">The Cost of Dismantling C</text>
  <text x="22" y="{top + plot_height / 2}" text-anchor="middle" font-size="15" transform="rotate(-90 22 {top + plot_height / 2})">Survived Robustness S</text>
</svg>
"""
    out.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "Data"
    parser = argparse.ArgumentParser(
        description="CoreHD dynamic dismantling for the ship network (pure-Python, no reinsertion)."
    )
    parser.add_argument("--edges", default=str(data_dir / "network_edges_ship.csv"))
    parser.add_argument("--source-col", default="node1")
    parser.add_argument("--target-col", default="node2")
    parser.add_argument("--weight-col", default="weight")
    parser.add_argument(
        "--step-fraction",
        type=float,
        default=0.1,
        help="Each round delete the top step_fraction of the current 2-core by "
        "degree (default 0.1, i.e. 10%% of remaining 2-core nodes per round).",
    )
    parser.add_argument(
        "--csize",
        type=int,
        default=1,
        help="Target component-size threshold for tree-breaking. "
        "1 (default) fully fragments the network so residual edges reach 0 "
        "(comparable to HDC/HBC). Use e.g. ceil(0.01*N) for the paper variant.",
    )
    parser.add_argument(
        "--normalization",
        choices=("sum", "minmax", "none"),
        default="sum",
        help="How to normalize edge weights before computing removal cost.",
    )
    parser.add_argument("--order-out", default="")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--curve-out", default="")
    parser.add_argument("--weighted-summary-out", default="")
    parser.add_argument("--plot-out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(__file__).resolve().parents[1] / "Result1.0"
    order_out = args.order_out or str(result_dir / "ship_corehd_dismantle_order_frac10.csv")
    summary_out = args.summary_out or str(result_dir / "ship_corehd_summary_frac10.json")
    curve_out = args.curve_out or str(result_dir / "ship_corehd_weighted_robustness_frac10.csv")
    weighted_summary_out = args.weighted_summary_out or str(
        result_dir / "ship_corehd_weighted_robustness_frac10.json"
    )
    plot_out = args.plot_out or str(result_dir / "ship_corehd_weighted_robustness_frac10.svg")

    edges, nodes = read_directed_edges(args.edges, args.source_col, args.target_col, args.weight_col)
    raw_undirected = build_undirected_weighted(edges)
    adjacency = build_adjacency(raw_undirected)
    num_nodes = len(nodes)

    order = corehd_dismantle_order(adjacency, step_fraction=args.step_fraction, csize=args.csize)
    full_order = complete_order(order, sorted(nodes))

    summary = {
        "num_nodes": num_nodes,
        "num_edges": len(raw_undirected),
        "step_fraction": args.step_fraction,
        "csize": args.csize,
        "initial_lcc": largest_component_size(set(adjacency), adjacency),
        "removed_until_terminal": len(order),
        "removed_fraction_until_terminal": len(order) / max(num_nodes, 1),
        "robustness": standard_robustness(adjacency, full_order, num_nodes),
    }

    edge_weights = normalize_weights(raw_undirected, args.normalization)
    rows, weighted_summary = compute_weighted_curve(edge_weights, order)
    weighted_summary["step_fraction"] = args.step_fraction
    weighted_summary["csize"] = args.csize

    write_order(order_out, order)
    Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_out).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_curve(curve_out, rows)
    Path(weighted_summary_out).write_text(
        json.dumps(weighted_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_plot(
        plot_out,
        rows,
        float(weighted_summary["weighted_robustness"]),
        title="CoreHD Weighted Robustness",
    )

    print(json.dumps({"standard": summary, "weighted": weighted_summary}, indent=2, ensure_ascii=False))
    print(f"order saved to {order_out}")
    print(f"summary saved to {summary_out}")
    print(f"curve saved to {curve_out}")
    print(f"weighted summary saved to {weighted_summary_out}")
    if plot_out:
        print(f"plot saved to {plot_out}")


if __name__ == "__main__":
    main()
