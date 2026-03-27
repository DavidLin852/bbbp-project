from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATBackbone(nn.Module):
    """
    Must match the backbone used in SMARTS pretrain.
    """
    def __init__(self, in_dim: int, hidden: int, heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()

        self.convs.append(GATConv(in_dim, hidden, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads, concat=True))
        self.convs.append(GATConv(hidden * heads, hidden, heads=1, concat=True))

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch_idx)
        return g
