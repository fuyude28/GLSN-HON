"""Compare saved critical-node rankings with one common CB-LDF rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from robustness_experiment import (
    DATA, GLSN, HERE, add_links, attack_schedule, evaluate, read_graph, write_csv,
)

CRITICAL = GLSN / "critical"
DYNAMIC_FILE = CRITICAL / "DND-DRL/results/source_data/baseline_curves.csv"
RANKING_SOURCES = {
    "DND-DRL": (CRITICAL / "DND-DRL/results/formal_removal_sequence.csv", None, "state_node"),
    "DND": (CRITICAL / "baselines/DND/results/dnd_curve.csv", None, "removed_state"),
    "FINDER": (CRITICAL / "FINDER/results/finder_curve.csv", None, "removed_state"),
    "TAD": (CRITICAL / "baselines/TAD/results/tad_curve.csv", None, "removed_state"),
    "CoreHD": (CRITICAL / "baselines/CoreHD/results/corehd_curve.csv", None, "removed_state"),
    "TotalDegree": (DYNAMIC_FILE, "dynamic_total_degree", "removed_state"),
    "InDegree": (DYNAMIC_FILE, "dynamic_in_degree", "removed_state"),
    "OutDegree": (DYNAMIC_FILE, "dynamic_out_degree", "removed_state"),
    "PageRank": (DYNAMIC_FILE, "dynamic_pagerank", "removed_state"),
    "ControlCentrality": (CRITICAL / "baselines/Control Centrality/results/control_centrality_curve.csv", None, "removed_state"),
}
METHODS = (*RANKING_SOURCES, "RandomNode")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def complete_order(order: list[int], n: int) -> list[int]:
    """Preserve the saved prefix; append unranked original IDs in ascending order."""
    seen = set(order)
    if len(seen) != len(order):
        raise ValueError("saved ranking contains duplicate states")
    if not seen.issubset(range(n)):
        raise ValueError("saved ranking contains states outside the input graph")
    return order + [node for node in range(n) if node not in seen]


def load_rankings(labels: list[str]) -> tuple[dict[str, list[int]], list[dict]]:
    index = {label: node for node, label in enumerate(labels)}
    tables = {path: read_csv(path) for path, _, _ in RANKING_SOURCES.values()}
    orders, sources = {}, []
    for method, (path, selector, column) in RANKING_SOURCES.items():
        rows = [row for row in tables[path]
                if (selector is None or row["method"] == selector) and row[column]]
        saved = [index[row[column]] for row in rows]
        if not saved:
            raise ValueError(f"empty saved ranking: {method}")
        orders[method] = complete_order(saved, len(labels))
        sources.append(dict(method=method, source=str(path.relative_to(GLSN)),
                            method_filter=selector or "", state_column=column,
                            saved_ranked_states=len(saved), completed_states=len(labels)))
    return orders, sources


def require_budget(added: list[tuple[int, int]], budget: int, method: str) -> None:
    if len(added) != budget:
        raise ValueError(f"{method}: only {len(added)}/{budget} links after scanning all states")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-results", type=Path, default=HERE / "results")
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=HERE / "results/ranking_ldf")
    args = parser.parse_args()
    if args.random_repeats < 2:
        parser.error("random-repeats must be at least two to estimate sample SD")
    reference = json.loads((args.reference_results / "protocol.json").read_text(encoding="utf-8"))
    budgets = reference["budgets"]
    labels, base_edges = read_graph(DATA / "network-ho.csv")
    n = len(labels)
    orders, sources = load_rankings(labels)
    source_by_method = {row["method"]: row for row in sources}
    attack_order = orders["DND-DRL"][:source_by_method["DND-DRL"]["saved_ranked_states"]]
    attacked, active = attack_schedule(n, attack_order, reference["attack_rounds"])
    baseline_auc, baseline_gscc = evaluate(n, base_edges, attacked, active)
    saved_baseline = read_csv(args.reference_results / "baseline_curve.csv")
    if (n != reference["nodes"] or len(base_edges) != reference["original_edges"]
            or attacked != [int(row["attacked"]) for row in saved_baseline]
            or baseline_gscc != [int(row["gscc"]) for row in saved_baseline]
            or not math.isclose(baseline_auc, reference["baseline_partial_auc"], abs_tol=1e-12)):
        raise ValueError("current network/attack baseline differs from the existing panel")
    reference_ldf = {int(row["budget"]): float(row["mean_delta_r"])
                     for row in read_csv(args.reference_results / "summary.csv")
                     if row["method"] == "CB-LDF"}
    reference_edges = [(row["source_state"], row["target_state"])
                       for row in read_csv(args.reference_results / "selected_edges.csv")
                       if row["method"] == "CB-LDF"]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "ranking_sources.csv", sources, list(sources[0]))
    trials, traces, curves, coverage = [], [], [], []
    for method in METHODS:
        repeats = args.random_repeats if method == "RandomNode" else 1
        for repeat in range(repeats):
            seed = args.seed + repeat
            if method == "RandomNode":
                order = random.Random(seed).sample(range(n), n)
                saved_length = n
            else:
                order = orders[method]
                saved_length = source_by_method[method]["saved_ranked_states"]
            added, trace = add_links(labels, base_edges, order, "CB-LDF",
                                     max(budgets), random.Random(seed))
            require_budget(added, max(budgets), method)
            if method == "DND-DRL" and [(labels[u], labels[v]) for u, v in added] != reference_edges:
                raise ValueError("DND-DRL-LDF links differ from existing CB-LDF links")
            coverage.append(dict(method=method, repeat=repeat, seed=seed,
                                 saved_ranked_states=saved_length,
                                 scanned_states=trace[-1]["priority_rank"],
                                 skipped_empty_states=trace[-1]["priority_rank"] - len(added),
                                 edges_added=len(added),
                                 tail_edges=sum(row["priority_rank"] > saved_length for row in trace)))
            for number, row in enumerate(trace, start=1):
                traces.append(dict(method=method, repeat=repeat, seed=seed, edge_number=number,
                                   from_unranked_tail=row["priority_rank"] > saved_length, **row))
            for budget in budgets:
                area, gscc = evaluate(n, base_edges + added[:budget], attacked, active)
                delta = (area - baseline_auc) / baseline_auc
                if method == "DND-DRL" and not math.isclose(delta, reference_ldf[budget], abs_tol=1e-12):
                    raise ValueError(f"DND-DRL-LDF does not reproduce existing CB-LDF at B={budget}")
                trials.append(dict(method=method, repeat=repeat, seed=seed, budget=budget,
                                   edges_added=budget, baseline_auc=baseline_auc, auc=area,
                                   delta_r=delta, final_gscc=gscc[-1]))
                curves.extend(dict(method=method, repeat=repeat, seed=seed, budget=budget,
                                   attacked=q, gscc=g) for q, g in zip(attacked, gscc))
            print(f"{method}-LDF {repeat + 1}/{repeats}: {len(added)} edges; "
                  f"scanned={trace[-1]['priority_rank']}; tail={coverage[-1]['tail_edges']}; "
                  f"delta_R(B={max(budgets)})={trials[-1]['delta_r']:.6f}", flush=True)
    grouped = defaultdict(list)
    for row in trials:
        grouped[row["method"], row["budget"]].append(row)
    summary = []
    for method in METHODS:
        for budget in budgets:
            group = grouped[method, budget]
            values = [row["delta_r"] for row in group]
            summary.append(dict(method=method, budget=budget, trials=len(group),
                                min_edges_added=min(row["edges_added"] for row in group),
                                mean_edges_added=statistics.mean(row["edges_added"] for row in group),
                                mean_delta_r=statistics.mean(values),
                                sd_delta_r=statistics.stdev(values) if len(values) > 1 else ""))
    for name, rows in [("trials", trials), ("selected_edges", traces),
                       ("gscc_curves", curves), ("coverage", coverage), ("summary", summary)]:
        write_csv(args.out / f"{name}.csv", rows, list(rows[0]))
    write_csv(args.out / "baseline_curve.csv",
              [dict(attacked=q, gscc=g) for q, g in zip(attacked, baseline_gscc)],
              ["attacked", "gscc"])
    protocol = dict(nodes=n, original_edges=len(base_edges), original_gscc=baseline_gscc[0],
                    baseline_partial_auc=baseline_auc, budgets=budgets,
                    ranking_methods=list(METHODS), deterministic_runs=1,
                    random_repeats=args.random_repeats, random_seeds=list(range(args.seed, args.seed + args.random_repeats)),
                    attack_rounds=reference["attack_rounds"], attack_batch=attacked[1], attack_horizon=attacked[-1],
                    attack_source=source_by_method["DND-DRL"]["source"],
                    attack_protocol=reference["attack_protocol"], area_definition=reference["area_definition"],
                    candidate_protocol=reference["candidate_protocol"],
                    priority_protocol="one pass over completed ranking; skip empty pools; one edge per state",
                    completion_protocol="preserve saved removal order; append missing original graph IDs in ascending order",
                    budget_protocol="nested prefixes of 400-edge sequence; fail if any run cannot reach the full budget",
                    link_selection="CB-LDF: minimize current out_degree(source)+in_degree(target); lexicographic edge-ID ties",
                    random_node_protocol="uniform permutation of all original state IDs; only node order is randomized",
                    dnd_ldf_matches_existing_panel=True,
                    limitations=["fixed DND-DRL attacker may favor DND-DRL-guided bypasses",
                                 "partial robustness over 2144 removals, not full dismantling AUC"])
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Saved {len(trials)} trials and {len(traces)} edges to {args.out}", flush=True)


if __name__ == "__main__":
    main()
