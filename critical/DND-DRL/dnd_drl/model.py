"""Sparse Dir-GNN encoder and state-action Q decoder."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .graph import DirectedGraph


def directed_neighbor_means(
    h: torch.Tensor, sources: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return incoming/outgoing means and their unnormalized degrees."""
    n = h.shape[0]
    incoming = torch.zeros_like(h).index_add(0, targets, h[sources])
    outgoing = torch.zeros_like(h).index_add(0, sources, h[targets])
    ones = torch.ones(sources.numel(), dtype=h.dtype, device=h.device)
    in_degree = h.new_zeros(n).index_add(0, targets, ones)
    out_degree = h.new_zeros(n).index_add(0, sources, ones)
    return (incoming / in_degree.clamp_min(1).unsqueeze(1),
            outgoing / out_degree.clamp_min(1).unsqueeze(1),
            in_degree, out_degree)


def normalized_residual_degrees(degree: torch.Tensor) -> torch.Tensor:
    """Log-scale each direction by its residual maximum, independent of graph size."""
    log_degree = torch.log1p(degree)
    return log_degree / log_degree.amax(dim=0).clamp_min(1e-12)


class DirQNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 64, message_passing_steps: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_passing_steps = message_passing_steps
        self.input_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.self_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim)
                                         for _ in range(message_passing_steps))
        self.in_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False)
                                       for _ in range(message_passing_steps))
        self.out_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False)
                                        for _ in range(message_passing_steps))
        self.decoder = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self, graph: DirectedGraph, removed: tuple[int, ...] | list[int] | set[int] = ()
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        active = torch.ones(graph.num_nodes, dtype=torch.bool, device=device)
        if removed:
            active[list(removed)] = False
        edges = graph.residual_edges(removed)
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long, device=device)
            sources, targets = edge_index[:, 0], edge_index[:, 1]
        else:
            sources = targets = torch.empty(0, dtype=torch.long, device=device)

        degree = torch.zeros((graph.num_nodes, 2), dtype=torch.float32, device=device)
        ones = torch.ones(len(edges), dtype=torch.float32, device=device)
        degree[:, 0].index_add_(0, targets, ones)
        degree[:, 1].index_add_(0, sources, ones)
        h = self.input_mlp(normalized_residual_degrees(degree)) * active.unsqueeze(1)
        for self_layer, in_layer, out_layer in zip(
            self.self_layers, self.in_layers, self.out_layers
        ):
            mean_in, mean_out, _, _ = directed_neighbor_means(h, sources, targets)
            h = F.relu(self_layer(h) + in_layer(mean_in) + out_layer(mean_out))
            h = h * active.unsqueeze(1)
        graph_state = h[active].mean(dim=0) if active.any() else h.new_zeros(self.hidden_dim)
        return h, graph_state, active

    def forward(
        self, graph: DirectedGraph, removed: tuple[int, ...] | list[int] | set[int] = ()
    ) -> torch.Tensor:
        nodes, state, active = self.encode(graph, removed)
        shared = state.unsqueeze(0).expand(graph.num_nodes, -1)
        q = self.decoder(torch.cat((nodes, shared, nodes * shared), dim=1)).squeeze(1)
        return q.masked_fill(~active, -1e30)

    def forward_batch(
        self, graphs: Sequence[DirectedGraph],
        removed_batch: Sequence[tuple[int, ...] | list[int] | set[int]],
    ) -> list[torch.Tensor]:
        """Score independent residual graphs with one disjoint-union forward pass."""
        device = next(self.parameters()).device
        offsets = [0]
        sources: list[int] = []
        targets: list[int] = []
        active_parts = []
        for graph, removed in zip(graphs, removed_batch):
            start = offsets[-1]
            active = torch.ones(graph.num_nodes, dtype=torch.bool)
            if removed:
                active[list(removed)] = False
            active_parts.append(active)
            for u, v in graph.residual_edges(removed):
                sources.append(start + u)
                targets.append(start + v)
            offsets.append(start + graph.num_nodes)
        active = torch.cat(active_parts).to(device)
        source = torch.tensor(sources, dtype=torch.long, device=device)
        target = torch.tensor(targets, dtype=torch.long, device=device)
        degree = torch.zeros((offsets[-1], 2), dtype=torch.float32, device=device)
        ones = torch.ones(len(sources), dtype=torch.float32, device=device)
        degree[:, 0].index_add_(0, target, ones)
        degree[:, 1].index_add_(0, source, ones)
        scaled = torch.log1p(degree)
        for begin, end in zip(offsets, offsets[1:]):
            scaled[begin:end] /= scaled[begin:end].amax(dim=0).clamp_min(1e-12)

        h = self.input_mlp(scaled) * active.unsqueeze(1)
        for self_layer, in_layer, out_layer in zip(
            self.self_layers, self.in_layers, self.out_layers
        ):
            mean_in, mean_out, _, _ = directed_neighbor_means(h, source, target)
            h = F.relu(self_layer(h) + in_layer(mean_in) + out_layer(mean_out))
            h = h * active.unsqueeze(1)

        graph_ids = torch.repeat_interleave(
            torch.arange(len(graphs), device=device),
            torch.tensor([g.num_nodes for g in graphs], device=device))
        graph_states = h.new_zeros((len(graphs), self.hidden_dim))
        graph_states.index_add_(0, graph_ids, h)
        counts = h.new_zeros(len(graphs)).index_add_(
            0, graph_ids, active.to(h.dtype)).clamp_min(1)
        graph_states = graph_states / counts.unsqueeze(1)
        shared = graph_states[graph_ids]
        q = self.decoder(torch.cat((h, shared, h * shared), dim=1)).squeeze(1)
        q = q.masked_fill(~active, -1e30)
        return [q[begin:end] for begin, end in zip(offsets, offsets[1:])]
