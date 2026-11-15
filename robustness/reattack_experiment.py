"""Replay the Original GLSN removal sequence on all augmented networks."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from bisect import bisect_left, insort
from collections import defaultdict
from pathlib import Path

import igraph as ig
from robustness_experiment import DATA, GLSN, HERE, write_csv, read_graph

OUT = HERE / "results/reattack"
BUDGETS = (200, 400)
RANDOM_SEEDS = tuple(range(42, 52))
SEQUENCE = OUT / "original_attack_sequence.csv"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_edge_sets(labels: list[str], base_edges: list[tuple[int, int]]) -> dict:
    index = {label: node for node, label in enumerate(labels)}
    groups = defaultdict(list)
    for row in read_csv(HERE / "results/ranking_ldf/selected_edges.csv"):
        if row["method"] == "DND-DRL" or (
                row["method"] == "RandomNode" and int(row["seed"]) in RANDOM_SEEDS):
            method = row["method"] + "-LDF"
            seed = int(row["seed"]) if row["method"] == "RandomNode" else 0
            groups[method, seed].append(row)
    expected = {("DND-DRL-LDF", 0)}
    expected.update(("RandomNode-LDF", seed) for seed in RANDOM_SEEDS)
    if set(groups) != expected:
        raise ValueError("missing prespecified augmentation method or RandomNode seed")
    result = {}
    for key, rows in groups.items():
        rows.sort(key=lambda row: int(row["edge_number"]))
        if [int(row["edge_number"]) for row in rows] != list(range(1, 401)):
            raise ValueError(f"incomplete 400-edge sequence: {key}")
        edges = [(index[row["source_state"]], index[row["target_state"]]) for row in rows]
        if len(set(edges)) != 400 or set(edges).intersection(base_edges):
            raise ValueError(f"duplicate or pre-existing augmentation edge: {key}")
        result[key] = edges
    return result


def trapezoid(gscc: list[int], q: list[int], n: int) -> float:
    return sum((a+b)*(right-left) for a,b,left,right in
               zip(gscc, gscc[1:], q, q[1:]))/(2*n*n)


def fixed_curve(n: int, edges: list[tuple[int, int]], order: list[int]) -> list[int]:
    topology = ig.Graph(n=n, edges=edges, directed=True)
    removed_ids = []
    sizes = [max(topology.connected_components(mode="strong").sizes(), default=0)]
    for original_id in order:
        current_id = original_id - bisect_left(removed_ids, original_id)
        topology.delete_vertices(current_id)
        insort(removed_ids, original_id)
        sizes.append(max(topology.connected_components(mode="strong").sizes(), default=0))
    return sizes


def main() -> None:
    labels, base_edges = read_graph(DATA / "network-ho.csv")
    n = len(labels)
    index = {label: i for i, label in enumerate(labels)}
    original_rows = read_csv(SEQUENCE)
    order = [index[r["state_node"]] for r in original_rows if r["state_node"]]
    if len(order) != len(set(order)):
        raise ValueError("duplicate node in shared Original GLSN sequence")
    sampled_q = list(range(0, 2145, math.ceil(n*0.01)))
    edge_sets = load_edge_sets(labels, base_edges)
    protocol = dict(graph="network-ho.csv", nodes=n, base_edges=len(base_edges),
                    budgets=list(BUDGETS), random_seeds=list(RANDOM_SEEDS),
                    methods=["Original GLSN", "DND-DRL-LDF", "RandomNode-LDF"],
                    attack="identical Original GLSN state sequence replayed on every network",
                    attack_source=str(SEQUENCE.relative_to(GLSN)),
                    sequence_length=len(order), attack_horizon=len(order),
                    attack_horizon_fraction=len(order)/n, partial_horizon=2144,
                    termination="end of shared original sequence; no graph-specific early stop",
                    reranking=False, retraining=False,
                    augmentation_sources=["robustness/results/ranking_ldf/selected_edges.csv"],
                    sequence_auc_definition="sum of post-removal GSCC/N divided by N, through shared sequence endpoint",
                    partial_auc_definition="trapezoid of GSCC/N versus removed/N at 0,134,...,2144",
                    uncertainty="sample SD across RandomNode seeds 42-51",
                    limitations=["fixed original-network attack; no adaptation to augmented structures",
                                 "shared horizon need not collapse augmented networks",
                                 "single empirical network and one original attack sequence"])
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    jobs = [("Original GLSN", 0, 0)]
    jobs.extend((method,budget,seed) for budget in BUDGETS
                for method in ("DND-DRL-LDF", "RandomNode-LDF")
                for seed in (RANDOM_SEEDS if method=="RandomNode-LDF" else (0,)))
    trials, sampled_rows = [], []
    (OUT / "runs").mkdir(exist_ok=True)
    for method, budget, seed in jobs:
        started = time.perf_counter()
        edges = base_edges if budget==0 else base_edges+edge_sets[method,seed][:budget]
        curve = fixed_curve(n, edges, order)
        if method=="Original GLSN":
            assert curve==[int(r["gscc"]) for r in original_rows], "original baseline curve changed"
        samples = [curve[q] for q in sampled_q]
        row = dict(method=method, budget=budget, seed=seed, nodes=n, edges=len(edges),
                   initial_gscc=curve[0], removals=len(order),
                   sequence_auc=sum(curve[1:])/(n*n),
                   partial_auc=trapezoid(samples,sampled_q,n),
                   gscc_at_2144=curve[2144], gscc_at_sequence_end=curve[-1],
                   elapsed_seconds=time.perf_counter()-started)
        baseline = row if not trials else trials[0]
        row["sequence_delta_r"] = row["sequence_auc"]/baseline["sequence_auc"]-1
        row["partial_delta_r"] = row["partial_auc"]/baseline["partial_auc"]-1
        trials.append(row)
        curve_rows = [dict(removed=q, state_node=labels[order[q-1]] if q else "",
                           gscc=g, gscc_fraction=g/n) for q,g in enumerate(curve)]
        stem=f"{method}_B{budget}_seed{seed}"
        write_csv(OUT / "runs" / f"{stem}.csv", curve_rows, list(curve_rows[0]))
        (OUT / "runs" / f"{stem}.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
        sampled_rows.extend(dict(method=method,budget=budget,seed=seed,removed=q,
                                 gscc=curve[q],gscc_fraction=curve[q]/n) for q in sampled_q)
        print(f"{stem}: common removals={len(order)}, end GSCC={curve[-1]}, "
              f"partial gain={row['partial_delta_r']:.4%}, time={row['elapsed_seconds']:.1f}s",flush=True)
    write_csv(OUT / "trials.csv", trials, list(trials[0]))
    write_csv(OUT / "sampled_curves.csv", sampled_rows, list(sampled_rows[0]))
    groups=defaultdict(list)
    for row in trials:
        groups[row["method"],row["budget"]].append(row)
    summary=[]
    for (method,budget),group in groups.items():
        row=dict(method=method,budget=budget,runs=len(group))
        for key in ("sequence_auc","sequence_delta_r","partial_auc","partial_delta_r",
                    "removals","gscc_at_2144","gscc_at_sequence_end"):
            values=[r[key] for r in group]
            row[f"mean_{key}"]=statistics.mean(values)
            row[f"sd_{key}"]=statistics.stdev(values) if len(values)>1 else ""
        summary.append(row)
    write_csv(OUT / "summary.csv",summary,list(summary[0]))
    (OUT / "baseline_reproduction.json").write_text(json.dumps(
        dict(exact_sequence_match=True,exact_gscc_match=True,
             reference="original_attack_sequence.csv",sequence_length=len(order),
             baseline_sequence_auc=trials[0]["sequence_auc"]),indent=2),encoding="utf-8")
    print(f"Saved {len(trials)} fixed-sequence experiments",flush=True)


if __name__ == "__main__":
    main()
