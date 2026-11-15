"""Paper SPL node features: out-degree, betweenness, total degree."""

from __future__ import annotations

import igraph as ig
import numpy as np
import torch
from torch_geometric.data import Data


def graph_features(graph: ig.Graph) -> Data:
    if not graph.is_directed():
        raise ValueError("DND requires a directed graph")
    values = np.column_stack((
        graph.degree(mode="out", loops=False),
        graph.betweenness(directed=True),
        graph.degree(mode="all", loops=False),
    )).astype(np.float32)
    # The article does not specify feature scaling; normalize each feature
    # within a graph so small training and large inference graphs are comparable.
    values = np.log1p(values)
    values /= np.maximum(values.max(axis=0, keepdims=True), 1e-12)
    edges = graph.get_edgelist()
    edge_index = (torch.tensor(edges, dtype=torch.long).T.contiguous()
                  if edges else torch.empty((2, 0), dtype=torch.long))
    return Data(x=torch.from_numpy(values), edge_index=edge_index)
