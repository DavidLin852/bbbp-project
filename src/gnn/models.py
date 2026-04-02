"""
GNN model definitions for BBB permeability prediction.

Models: GCN, GIN, GAT
Architecture: Message-passing backbone + global pooling + task-specific head

Graph featurization (from src.features.graph):
- Node features: atomic number (one-hot), degree, H-count, formal charge,
  hybridization (one-hot), aromaticity, scaled mass  -> 39 dimensions
- Edge features: bond type (one-hot), conjugation, ring membership  -> 7 dimensions

Design choices:
- Global mean pooling (standard for graph classification/regression)
- Batch normalization after each message-passing layer
- Dropout for regularization (critical on small datasets like B3DB)
- Classification: BCE loss (binary), sigmoid output
- Regression: MSE loss, linear output
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GATConv,
    global_mean_pool,
    BatchNorm,
)


# ==================== Configuration ====================

@dataclass(frozen=True)
class GNNConfig:
    """
    Configuration for GNN models.

    Attributes:
        hidden_dim: Hidden dimension for message passing (default: 128)
        num_layers: Number of message-passing layers (default: 3)
        dropout: Dropout rate (default: 0.3)
        lr: Learning rate (default: 1e-3)
        weight_decay: Weight decay (default: 1e-4)
        epochs: Maximum training epochs (default: 300)
        patience: Early stopping patience (default: 30)
        batch_size: Batch size (default: 64)
        min_atoms: Minimum atoms per molecule (default: 1)
    """
    hidden_dim: int = 128
    num_layers: int = 3
    heads: int = 4  # Number of attention heads for GAT
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 30
    batch_size: int = 64
    min_atoms: int = 1


# ==================== Backbones ====================

class GCN(nn.Module):
    """
    Graph Convolutional Network.

    Uses GCNConv layers with batch normalization and ELU activation.
    Node features only (edge features averaged into source node messages).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer: input -> hidden
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))

        # Output layer: hidden -> hidden
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch_idx)
        return g


class GIN(nn.Module):
    """
    Graph Isomorphism Network.

    Uses GINConv layers (MLP + sum aggregation) with batch normalization.
    Theoretically more expressive than GCN (WKSubgraph isomorphism).
    Node features only.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer: input -> hidden
        mlp1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(nn=mlp1, train_eps=False))
        self.bns.append(BatchNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            self.bns.append(BatchNorm(hidden_dim))

        # Output layer
        mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(nn=mlp_out, train_eps=False))
        self.bns.append(BatchNorm(hidden_dim))

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch_idx)
        return g


class GAT(nn.Module):
    """
    Graph Attention Network.

    Uses GATConv layers with multi-head attention.
    Attention mechanism allows the model to weight neighboring nodes
    differently, capturing heterogeneous importance.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer: input -> hidden (multi-head)
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True))
        self.bns.append(BatchNorm(hidden_dim * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
            )
            self.bns.append(BatchNorm(hidden_dim * heads))

        # Output layer: single head, not concatenated
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False))
        self.bns.append(BatchNorm(hidden_dim))

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        # Note: edge_attr is available in batch.edge_attr but not used by GATConv
        # This is a design choice for consistency across GCN/GIN/GAT

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)  # GATConv ignores edge_attr by default
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch_idx)
        return g


# ==================== Task Heads ====================

class GNNClassificationHead(nn.Module):
    """Classification head for GNN backbone."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, g):
        return self.head(g)


class GNNRegressionHead(nn.Module):
    """Regression head for GNN backbone."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, g):
        return self.head(g)


# ==================== Full Models ====================

def build_gnn(
    model_type: str,
    in_dim: int,
    config: GNNConfig | None = None,
) -> nn.Module:
    """
    Build a GNN model by type.

    Args:
        model_type: One of "gcn", "gin", "gat"
        in_dim: Input node feature dimension
        config: GNN configuration

    Returns:
        Full GNN model with classification head
    """
    cfg = config or GNNConfig()
    hidden = cfg.hidden_dim
    layers = cfg.num_layers
    dropout = cfg.dropout

    if model_type == "gcn":
        backbone = GCN(in_dim, hidden, layers, dropout)
    elif model_type == "gin":
        backbone = GIN(in_dim, hidden, layers, dropout)
    elif model_type == "gat":
        backbone = GAT(in_dim, hidden, layers, dropout)
    else:
        raise ValueError(f"Unknown GNN type: {model_type}")

    return backbone
