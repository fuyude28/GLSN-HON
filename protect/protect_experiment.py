"""Gu-inspired, renewed-per-round protection on a directed higher-order network."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import igraph as ig
import torch

from protection_rankings import CRITICAL, METHODS, DirectedGraph, ProtectionRankings

HERE = Path(__file__).resolve().parent


def gscc_size(live: ig.Graph) -> int:
    # Failed vertices retain their IDs as isolated placeholders. While a
    # nontrivial SCC remains this has no effect on the maximum SCC size.
    return max(live.connected_components(mode='strong').sizes(), default=0)


def gscc_candidates(live: ig.Graph) -> list[int]:
    """Same action mask as DirectedGraph.gscc_state, including tied largest SCCs."""
    components = live.connected_components(mode='strong')
    sizes = components.sizes()
    largest = max(sizes, default=0)
    if largest <= 1:
        return []
    return [node for node, component in enumerate(components.membership)
            if sizes[component] == largest]


def random_candidates(rng: random.Random, n: int, pool: list[int], batch: int) -> list[int]:
    """Uniform sample before protection filtering, with shared per-round priorities."""
    eligible = set(pool)
    order = rng.sample(range(n), n)
    return [node for node in order if node in eligible][:batch]


def run_trial(graph: DirectedGraph, rankings, method: str, mode: str,
              seed: int, batch_size: int) -> tuple[list[dict], list[dict], dict]:
    live = ig.Graph(n=graph.num_nodes, edges=graph.edges, directed=True)
    initial = gscc_size(live)
    size = initial
    removed: set[int] = set()
    attack_rng = random.Random(seed)
    protection_rng = random.Random(seed + 10000)
    curve = [dict(round=0, attacked=0, protected=0, removed_node='',
                  gscc=initial, gscc_fraction=1.0, attack_fraction=0.0, cumulative_audc=0.0)]
    rounds = []
    audc = 0.0
    step = 0
    start = time.perf_counter()
    while size > 1:
        pool = gscc_candidates(live)
        if len(pool) <= batch_size:
            break
        step += 1
        size_before = size
        attacker_order = (rankings.rank('DND-DRL', live, removed, pool)
                          if mode == 'targeted' else None)
        if method == 'No protection':
            protected = set()
        elif method == 'Random protection':
            protected = set(protection_rng.sample(pool, batch_size))
        else:
            order = (attacker_order if method == 'DND-DRL' and mode == 'targeted'
                     else rankings.rank(method, live, removed, pool))
            protected = set(order[:batch_size])

        # Protection is replaced each round. Random candidates are drawn
        # before filtering; blocked candidates are not replaced. Targeted
        # attack follows the paper's explicit instruction to skip protection.
        candidates = (random_candidates(attack_rng, graph.num_nodes, pool, batch_size)
                      if mode == 'random' else
                      [node for node in attacker_order if node not in protected][:batch_size])
        failures = [node for node in candidates if node not in protected]
        applied = []
        for node in failures:
            live.delete_edges(live.incident(node, mode='all'))
            removed.add(node)
            applied.append(node)
            size = gscc_size(live)
            audc += size / (initial * graph.num_nodes)
            curve.append(dict(round=step, attacked=len(removed), protected=len(protected),
                              removed_node=node, gscc=size, gscc_fraction=size / initial,
                              attack_fraction=len(removed) / graph.num_nodes,
                              cumulative_audc=audc))
            if size <= 1:
                break
        rounds.append(dict(round=step, gscc_before=size_before, gscc_candidates_before=len(pool),
                           protected_nodes=json.dumps(sorted(protected)),
                           attack_candidates=json.dumps(candidates),
                           removed_nodes=json.dumps(applied),
                           attacked_after=len(removed), gscc_after=size))
    effective_count = sum(degree > 0 for degree in live.degree(mode='all'))
    terminal = dict(method=method, mode=mode, seed=seed, rounds=step,
                    attacks=len(removed), terminal_gscc=size,
                    terminal_gscc_fraction=size / initial,
                    terminal_attack_fraction=len(removed) / graph.num_nodes,
                    terminal_effective_nodes=effective_count,
                    terminal_gscc_candidates=len(gscc_candidates(live)),
                    stop_reason='gscc_collapsed' if size <= 1 else 'gscc_candidates_at_most_batch',
                    audc=audc, elapsed_seconds=time.perf_counter() - start)
    return curve, rounds, terminal


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


_WORKER_GRAPH = None
_WORKER_RANKINGS = None


def init_worker(graph_path: str, methods: list[str]) -> None:
    global _WORKER_GRAPH, _WORKER_RANKINGS
    torch.set_num_threads(1)
    _WORKER_GRAPH = DirectedGraph.read_csv_edge_table(graph_path)
    _WORKER_RANKINGS = ProtectionRankings(_WORKER_GRAPH, methods)


def execute_trial(job: tuple) -> dict:
    method, mode, seed, batch, out_dir = job
    curve, rounds, terminal = run_trial(_WORKER_GRAPH, _WORKER_RANKINGS,
                                       method, mode, seed, batch)
    prefix = f'{mode}_{method.lower().replace(" ", "_")}_{seed}'
    write_csv(Path(out_dir) / 'runs' / (prefix + '_curve.csv'),
              [dict(mode=mode, method=method, seed=seed, **row) for row in curve])
    if rounds:
        write_csv(Path(out_dir) / 'runs' / (prefix + '_rounds.csv'), rounds)
    terminal['run_prefix'] = prefix
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, default=CRITICAL.parent.parent / 'data/network-ho.csv')
    parser.add_argument('--random-repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--out-dir', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    if args.random_repeats < 1 or args.workers < 1:
        parser.error('random-repeats and workers must be positive')
    graph = DirectedGraph.read_csv_edge_table(args.graph)
    batch = math.floor(0.01 * graph.num_nodes) + 1
    seeds = list(range(args.seed, args.seed + args.random_repeats))
    jobs = [(method, mode, seed, batch, str(args.out_dir))
            for mode in ('targeted', 'random') for method in args.methods
            for seed in (seeds if mode == 'random' or method == 'Random protection' else seeds[:1])]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    terminals = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(args.graph), args.methods)) as pool:
        pending = {pool.submit(execute_trial, job): job for job in jobs}
        for future in as_completed(pending):
            row = future.result()
            terminals.append(row)
            print(f'{len(terminals)}/{len(jobs)} {row["mode"]} seed={row["seed"]} '
                  f'{row["method"]}: rounds={row["rounds"]} attacked={row["attacks"]} '
                  f'GSCC={row["terminal_gscc"]} AUDC={row["audc"]:.6f} '
                  f'{row["stop_reason"]}', flush=True)
    terminals.sort(key=lambda row: (row['mode'], METHODS.index(row['method']), row['seed']))
    write_csv(args.out_dir / 'protection_trials.csv', terminals)
    with (args.out_dir / 'protection_curves.csv').open('w', encoding='utf-8', newline='') as out:
        writer = None
        for terminal in terminals:
            path = args.out_dir / 'runs' / (terminal['run_prefix'] + '_curve.csv')
            with path.open(encoding='utf-8', newline='') as handle:
                reader = csv.DictReader(handle)
                if writer is None:
                    writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)
    metadata = dict(
        protocol='Gu-inspired attack-protect adaptation to directed higher-order states',
        reference_doi='10.1038/s42256-025-01070-2',
        nodes=graph.num_nodes, edges=len(graph.edges), initial_gscc=graph.gscc_size(),
        batch_size=batch, batch_rule='floor(0.01 * initial_N) + 1, matching public HDA/CI code',
        protection_methods=args.methods, random_seeds=seeds,
        protection_rule='replace protection set with current top batch each round; no cumulative immunity',
        targeted_attacker='DND-DRL rescored on residual graph; rank current GSCC candidates and skip protected states',
        random_attacker='uniform candidates from current GSCC before removing protected candidates; no replacement',
        common_random_numbers='same seed and full-node priority permutation per round across methods',
        effective_nodes='active states with at least one incident surviving directed edge',
        stop_rule='GSCC <= 1 or current GSCC candidate count <= batch_size',
        dynamic_methods=['DND-DRL', 'FINDER', 'Dynamic total degree', 'Dynamic in-degree',
                         'Dynamic out-degree', 'Dynamic PageRank'],
        static_methods=['TAD', 'CoreHD', 'DND', 'Control Centrality'],
        ranking_candidates='current largest nontrivial SCC; union of all tied largest SCCs, as in dismantling',
        protection_candidates='same current GSCC mask for every method, including random protection',
        attack_candidates='same current GSCC mask for targeted and random attacks',
        batch_semantics='freeze current GSCC candidates at round start; apply preselected batch without reselection',
        measurement='GSCC measured after each actual node removal',
        audc='sum(GSCC_after_each_removal) / (initial_GSCC * initial_N)',
        display='normalized GSCC; means across the declared seeds; no selected or mixed seeds',
        adaptation='directed GSCC replaces multiplex LMCC; original static baseline scoring conventions retained')
    (args.out_dir / 'protection_summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
