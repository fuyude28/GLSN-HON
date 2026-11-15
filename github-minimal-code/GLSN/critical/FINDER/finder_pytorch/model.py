from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .graph import FinderGraph


NEG_INF = -1.0e30


class FinderQNetwork(nn.Module):
    """GraphSAGE-style FINDER Q network for ordinary ND graphs."""

    def __init__(
        self,
        hidden_dim: int = 64,
        message_passing_steps: int = 3,
        aux_dim: int = 4,
        decoder_hidden_dim: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_passing_steps = message_passing_steps
        self.aux_dim = aux_dim

        self.input_proj = nn.Linear(2, hidden_dim)
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.update_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.cross_product = nn.Parameter(torch.empty(hidden_dim))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + aux_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.cross_product, std=0.01)

    def forward(
        self,
        graph: FinderGraph,
        removed: Sequence[int] | set[int] = (),
        mask_removed: bool = True,
    ) -> torch.Tensor:
        return self.q_values(graph, removed, mask_removed=mask_removed)

    def q_values(
        self,
        graph: FinderGraph,
        removed: Sequence[int] | set[int] = (),
        mask_removed: bool = True,
    ) -> torch.Tensor:
        device = self.cross_product.device
        removed_set = set(int(node) for node in removed)
        node_embed, aux = self.encode(graph, removed_set, device)
        state_embed = self.graph_embedding(node_embed, graph, removed_set)
        state_expand = state_embed.unsqueeze(0).expand(graph.num_nodes, -1)
        interaction = node_embed * state_expand * self.cross_product.unsqueeze(0)
        decoder_in = torch.cat([interaction, aux.expand(graph.num_nodes, -1)], dim=1)
        q = self.decoder(decoder_in).squeeze(-1)
        if mask_removed:
            invalid = set(range(graph.num_nodes)) - set(graph.candidate_nodes(removed_set))
            if invalid:
                q[list(sorted(invalid))] = NEG_INF
        return q

    def score_actions(
        self,
        graphs: Sequence[FinderGraph],
        removed_batch: Sequence[Sequence[int] | set[int]],
        actions: Sequence[int],
    ) -> torch.Tensor:
        values = []
        for graph, removed, action in zip(graphs, removed_batch, actions):
            q = self.q_values(graph, removed, mask_removed=False)
            values.append(q[int(action)])
        return torch.stack(values)

    def max_next_q(
        self,
        graphs: Sequence[FinderGraph],
        removed_batch: Sequence[Sequence[int] | set[int]],
    ) -> torch.Tensor:
        values = []
        for graph, removed in zip(graphs, removed_batch):
            candidates = graph.candidate_nodes(removed)
            if not candidates:
                values.append(torch.zeros((), device=self.cross_product.device))
            else:
                values.append(self.q_values(graph, removed)[candidates].max())
        return torch.stack(values)

    def encode(
        self, graph: FinderGraph, removed_set: set[int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.node_features(graph, removed_set, device)
        h = F.relu(self.input_proj(features))
        h = F.normalize(h, p=2, dim=1)
        adjacency = self.adjacency_matrix(graph, removed_set, device)
        for _ in range(self.message_passing_steps):
            neigh = torch.sparse.mm(adjacency, h)
            neigh = self.neighbor_proj(neigh)
            self_part = self.self_proj(h)
            h = F.relu(self.update_proj(torch.cat([neigh, self_part], dim=1)))
            h = F.normalize(h, p=2, dim=1)
        return h, self.aux_features(graph, removed_set, device)

    @staticmethod
    def node_features(graph: FinderGraph, removed_set: set[int], device: torch.device) -> torch.Tensor:
        features = torch.ones((graph.num_nodes, 2), dtype=torch.float32, device=device)
        for node in removed_set:
            features[node, 0] = 0.0
        return features

    @staticmethod
    def adjacency_matrix(graph: FinderGraph, removed_set: set[int], device: torch.device) -> torch.Tensor:
        rows = []
        cols = []
        for u, v in graph.edges:
            if u in removed_set or v in removed_set:
                continue
            rows.extend((u, v))
            cols.extend((v, u))
        indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
        values = torch.ones(len(rows), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values,
                                       (graph.num_nodes, graph.num_nodes)).coalesce()

    @staticmethod
    def aux_features(graph: FinderGraph, removed_set: set[int], device: torch.device) -> torch.Tensor:
        removed_ratio = len(removed_set) / max(graph.num_nodes, 1)
        residual_edges = len(graph.residual_edges(removed_set))
        removed_edges = len(graph.edges) - residual_edges
        removed_edge_ratio = removed_edges / max(len(graph.edges), 1)
        two_hop = 0
        adj = graph.adjacency()
        active = set(range(graph.num_nodes)) - removed_set
        for node in active:
            active_degree = sum(1 for nbr in adj[node] if nbr in active)
            two_hop += active_degree * (active_degree - 1) // 2
        two_hop_ratio = two_hop / max(graph.num_nodes * graph.num_nodes, 1)
        return torch.tensor(
            [[removed_ratio, removed_edge_ratio, two_hop_ratio, 1.0]],
            dtype=torch.float32,
            device=device,
        )

    @staticmethod
    def graph_embedding(
        node_embed: torch.Tensor, graph: FinderGraph, removed_set: set[int]
    ) -> torch.Tensor:
        candidates = graph.candidate_nodes(removed_set)
        if not candidates:
            active = graph.active_nodes(removed_set)
            if not active:
                return torch.zeros(node_embed.shape[1], dtype=node_embed.dtype, device=node_embed.device)
            return node_embed[active].sum(dim=0)
        return node_embed[candidates].sum(dim=0)
