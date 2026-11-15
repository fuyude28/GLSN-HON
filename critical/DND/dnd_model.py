"""Directed GAT classifier following the SPL configuration in DND Table I."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class DNDModel(nn.Module):
    def __init__(self, input_dim: int = 3,
                 channels: tuple[int, ...] = (40, 30, 20, 10),
                 heads: int = 5,
                 mlp_channels: tuple[int, ...] = (50, 50, 30)):
        super().__init__()
        widths = (input_dim, *channels)
        self.convs = nn.ModuleList(
            GATConv(widths[i], widths[i + 1], heads=heads, concat=False,
                    add_self_loops=True)
            for i in range(len(channels)))
        self.residuals = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1], bias=False)
            for i in range(len(channels)))
        dense_widths = (channels[-1], *mlp_channels, 1)
        self.dense = nn.ModuleList(
            nn.Linear(dense_widths[i], dense_widths[i + 1])
            for i in range(len(dense_widths) - 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, residual in zip(self.convs, self.residuals):
            h = F.elu(conv(h, edge_index) + residual(h))
        for layer in self.dense[:-1]:
            h = F.elu(layer(h))
        return torch.sigmoid(self.dense[-1](h)).squeeze(-1)
