"""Load the pretrained DND variant shipped with the TAD comparison code."""

from __future__ import annotations

import igraph as ig
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATConv


class ResidualGATLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int):
        super().__init__()
        self.gat = GATConv((in_channels, in_channels), out_channels, heads=heads,
                           dropout=0.05, concat=False)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.05)
        self.residual = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.gat(x, edge_index))) + self.residual(x)


class PretrainedDND(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (4, 40, 30, 20, 10)
        self.layers = nn.ModuleList(
            [ResidualGATLayer(channels[i], channels[i + 1], 5)
             for i in range(4)] + [GATConv((10, 10), 100, heads=1, concat=False)])
        self.mlp = nn.Sequential(nn.Linear(100, 100), nn.ELU(),
                                 nn.Linear(100, 50), nn.ELU(),
                                 nn.Linear(50, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.mlp(x).squeeze(-1)


def score_pretrained(graph, checkpoint) -> np.ndarray:
    """TAD implementation features: betweenness, in/out-degree, PageRank."""
    torch.set_num_threads(min(4, torch.get_num_threads()))
    original = ig.Graph(n=graph.num_nodes,
                        edges=[(u, v) for u, v in graph.edges if u != v],
                        directed=True)
    n = graph.num_nodes
    betweenness = np.asarray(original.betweenness(directed=True)) / max((n - 1) * (n - 2), 1)
    topology = nx.DiGraph()
    topology.add_nodes_from(range(n))
    topology.add_edges_from(original.get_edgelist())
    pagerank = nx.pagerank(topology, alpha=0.85)
    values = np.column_stack((betweenness, original.degree(mode="in"),
                              original.degree(mode="out"),
                              [pagerank[node] for node in range(n)]))
    edge_index = torch.tensor(original.get_edgelist(), dtype=torch.long).T.contiguous()
    model = PretrainedDND()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(values, dtype=torch.float32),
                     edge_index).numpy()
