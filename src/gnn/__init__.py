"""
GNN models for BBB permeability prediction.

This module provides supervised graph neural network baselines:
- GCN (Graph Convolutional Network)
- GIN (Graph Isomorphism Network)
- GAT (Graph Attention Network)

All models use the same backbone architecture with task-specific heads
for classification and regression.

Design notes:
- Models operate directly on molecular graphs (RDKit atom/bond features)
- No pretrained weights; fully supervised training from scaffold splits
- Consistent with the classical baseline pipeline (same splits, same metrics)
"""

from .models import (
    GCN,
    GIN,
    GAT,
    GNNClassificationHead,
    GNNRegressionHead,
    GNNConfig,
)

__all__ = [
    "GCN",
    "GIN",
    "GAT",
    "GNNClassificationHead",
    "GNNRegressionHead",
    "GNNConfig",
]
