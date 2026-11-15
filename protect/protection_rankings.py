"""Protection rankings on the current residual directed GLSN."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CRITICAL = HERE.parent / 'critical'
sys.path[:0] = [str(CRITICAL / 'DND-DRL'), str(CRITICAL / 'FINDER')]
from dnd_drl.agent import DNDDrLAgent
from dnd_drl.graph import DirectedGraph
from finder_pytorch.agent import FinderDQNAgent
from finder_pytorch.graph import FinderGraph

METHODS = (
    'DND-DRL', 'TAD', 'CoreHD', 'Dynamic total degree',
    'Dynamic in-degree', 'Dynamic out-degree', 'Dynamic PageRank',
    'FINDER', 'DND', 'Control Centrality', 'Random protection', 'No protection',
)
STATIC_FILES = {
    'TAD': ('baselines/TAD/results/tad_order.csv', None),
    'CoreHD': ('baselines/CoreHD/results/corehd_order.csv', None),
    'DND': ('baselines/DND/results/dnd_scores.csv', 'score'),
    'Control Centrality': ('baselines/Control Centrality/results/control_centrality_scores.csv',
                          'control_centrality'),
}


class ProtectionRankings:
    """Keep each baseline's existing static/dynamic scoring convention.

    Static baselines use their complete all-node rankings, rather than the
    truncated sequence of nodes deleted in a different experiment.
    Dynamic methods are rescored once at the start of every attack round.
    The caller supplies the current GSCC action mask used by dismantling.
    Neural networks still encode the entire residual graph, as in training.
    """
    def __init__(self, graph: DirectedGraph, methods: list[str]):
        self.graph = graph
        self.static = {}
        self.dnd = None
        self.finder = None
        self.projection = None
        for method in methods:
            if method not in STATIC_FILES:
                continue
            filename, score_col = STATIC_FILES[method]
            with (CRITICAL / filename).open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            scores = {graph.node_mapping[row['state_node']]:
                      float(row[score_col]) if score_col else -int(row['rank']) for row in rows}
            if set(scores) != set(range(graph.num_nodes)):
                raise ValueError(f'{method} ranking does not cover this graph')
            self.static[method] = scores

    def rank(self, method: str, live, removed: set[int], candidates: list[int]) -> list[int]:
        if method in self.static:
            scores = self.static[method]
        elif method == 'DND-DRL':
            if self.dnd is None:
                self.dnd = DNDDrLAgent.load(CRITICAL / 'DND-DRL/models/formal_best.pt', device='cpu')
                self.dnd.online.eval()
            with torch.inference_mode():
                scores = self.dnd.online(self.graph, removed).numpy()
        elif method == 'FINDER':
            if self.finder is None:
                self.finder = FinderDQNAgent.load(
                    str(CRITICAL / 'FINDER/models/finder_nd_ba_20k.pt'), device='cpu')
                self.finder.online.eval()
                self.projection = FinderGraph.from_edges(self.graph.num_nodes, self.graph.edges)
            with torch.inference_mode():
                scores = self.finder.online.q_values(
                    self.projection, removed, mask_removed=False).numpy()
        elif method == 'Dynamic PageRank':
            # Removed vertices are kept only as isolated placeholders in live.
            active = [v for v in range(self.graph.num_nodes) if v not in removed]
            residual = live.induced_subgraph(active)
            scores = dict(zip(active, residual.pagerank(directed=True, damping=0.85)))
        else:
            direction = {'Dynamic total degree': 'all', 'Dynamic in-degree': 'in',
                         'Dynamic out-degree': 'out'}[method]
            scores = live.degree(mode=direction)
        return sorted(candidates, key=lambda node: (-float(scores[node]), node))
