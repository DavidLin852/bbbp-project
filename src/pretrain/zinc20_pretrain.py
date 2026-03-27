"""
ZINC20 Self-Supervised Pretraining Models

Multi-task pretraining strategies:
1. Context Prediction: Predict local atom environment from node embeddings
2. Property Prediction: Predict molecular properties (logP, TPSA, MW, etc.)
3. Graph Masking: Reconstruct masked node/edge features

Based on:
- InfoGraph (Sun et al., 2019)
- Graph Multitask Training (Hu et al., 2019)
- Masked Graph Modeling (MaskGNN, 2022)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, global_max_pool


class GATBackbone(nn.Module):
    """Shared GAT backbone for all pretraining tasks"""
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        pool: str = "mean"
    ):
        super().__init__()
        self.dropout = dropout
        self.pool = pool

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(in_dim, hidden, heads=heads, concat=True))
        self.norms.append(nn.BatchNorm1d(hidden * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads, concat=True))
            self.norms.append(nn.BatchNorm1d(hidden * heads))

        # Final layer (no concatenation)
        self.convs.append(GATConv(hidden * heads, hidden, heads=1, concat=True))
        self.norms.append(nn.BatchNorm1d(hidden))

        # Output dimension
        self.out_dim = hidden

    def forward(self, batch, return_node_embeddings: bool = False):
        """
        Args:
            batch: PyG batch object
            return_node_embeddings: If True, return (graph_emb, node_emb)

        Returns:
            graph_embedding: [batch_size, hidden]
            node_embeddings: [num_nodes, hidden] (optional)
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling
        if self.pool == "mean":
            g = global_mean_pool(x, batch_idx)
        elif self.pool == "add":
            g = global_add_pool(x, batch_idx)
        elif self.pool == "max":
            g = global_max_pool(x, batch_idx)
        else:
            g = global_mean_pool(x, batch_idx)

        if return_node_embeddings:
            return g, x
        return g


class ContextPredictionHead(nn.Module):
    """
    Context Prediction: Predict atom types in neighborhood

    For each atom, predict which atom types (C, N, O, etc.) appear
    within k hops of it.
    """
    def __init__(self, hidden_dim: int, num_atom_types: int = 9):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_atom_types)
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: [num_nodes, hidden_dim]

        Returns:
            logits: [num_nodes, num_atom_types]
        """
        return self.head(node_embeddings)


class PropertyPredictionHead(nn.Module):
    """
    Property Prediction: Predict molecular properties

    Predicts 9 properties: logP, TPSA, MW, rotatable bonds,
    HBD, HBA, rings, fraction Csp3, aromatic proportion
    """
    def __init__(self, hidden_dim: int, num_props: int = 9):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_props)
        )

    def forward(self, graph_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_embeddings: [batch_size, hidden_dim]

        Returns:
            props: [batch_size, num_props]
        """
        return self.head(graph_embeddings)


class MaskedReconstructionHead(nn.Module):
    """
    Masked Reconstruction: Reconstruct masked node features

    Similar to BERT but for graphs - predict original features
    of masked nodes.
    """
    def __init__(self, hidden_dim: int, node_feat_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, node_feat_dim)
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: [num_nodes, hidden_dim]

        Returns:
            reconstructed: [num_nodes, node_feat_dim]
        """
        return self.head(node_embeddings)


@dataclass
class PretrainConfig:
    """Configuration for ZINC20 pretraining"""
    # Model architecture
    in_dim: int = 29  # Atom feature dim
    hidden: int = 128
    heads: int = 4
    num_layers: int = 3
    dropout: float = 0.2
    pool: str = "mean"

    # Task weights
    lambda_context: float = 1.0
    lambda_property: float = 1.0
    lambda_mask: float = 0.5

    # Context prediction
    num_atom_types: int = 9  # C, N, O, F, P, S, Cl, Br, I

    # Property prediction
    num_props: int = 9

    # Masking
    mask_ratio: float = 0.15  # Fraction of nodes to mask


class ZINC20PretrainModel(nn.Module):
    """
    Multi-task ZINC20 Pretraining Model

    Combines three pretraining objectives:
    1. Context Prediction (node-level)
    2. Property Prediction (graph-level)
    3. Masked Feature Reconstruction (node-level)
    """

    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.cfg = cfg

        # Shared backbone
        self.backbone = GATBackbone(
            in_dim=cfg.in_dim,
            hidden=cfg.hidden,
            heads=cfg.heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pool=cfg.pool
        )

        # Task-specific heads
        self.context_head = ContextPredictionHead(
            hidden_dim=cfg.hidden,
            num_atom_types=cfg.num_atom_types
        )

        self.property_head = PropertyPredictionHead(
            hidden_dim=cfg.hidden,
            num_props=cfg.num_props
        )

        self.mask_head = MaskedReconstructionHead(
            hidden_dim=cfg.hidden,
            node_feat_dim=cfg.in_dim
        )

    def forward(
        self,
        batch,
        mask_indices: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with all tasks

        Args:
            batch: PyG batch with x, edge_index, batch, context, props
            mask_indices: Boolean tensor [num_nodes] indicating masked nodes
            return_components: If True, return individual losses

        Returns:
            Dict with:
                - loss: Total weighted loss
                - context_loss: Context prediction loss
                - property_loss: Property prediction loss
                - mask_loss: Masked reconstruction loss
        """
        # Get embeddings
        graph_emb, node_emb = self.backbone(batch, return_node_embeddings=True)

        losses = {}
        total_loss = 0.0

        # 1. Context Prediction
        if batch.context is not None and self.cfg.lambda_context > 0:
            context_logits = self.context_head(node_emb)
            context_loss = F.binary_cross_entropy_with_logits(
                context_logits,
                batch.context,
                reduction='mean'
            )
            losses['context_loss'] = context_loss
            total_loss += self.cfg.lambda_context * context_loss

        # 2. Property Prediction
        if batch.props is not None and self.cfg.lambda_property > 0:
            # Ensure graph_emb is [batch_size, hidden]
            if graph_emb.dim() != 2:
                graph_emb = graph_emb.view(-1, self.cfg.hidden)

            prop_logits = self.property_head(graph_emb)

            # Ensure batch.props is [batch_size, num_props]
            if batch.props.dim() == 1:
                batch_props = batch.props.view(-1, self.cfg.num_props)
            else:
                batch_props = batch.props

            property_loss = F.mse_loss(prop_logits, batch_props, reduction='mean')
            losses['property_loss'] = property_loss
            total_loss += self.cfg.lambda_property * property_loss

        # 3. Masked Reconstruction
        if mask_indices is not None and self.cfg.lambda_mask > 0:
            mask_logits = self.mask_head(node_emb[mask_indices])
            mask_targets = batch.x[mask_indices]
            mask_loss = F.mse_loss(mask_logits, mask_targets, reduction='mean')
            losses['mask_loss'] = mask_loss
            total_loss += self.cfg.lambda_mask * mask_loss

        losses['loss'] = total_loss

        if return_components:
            return losses
        return total_loss

    def generate_mask(self, batch: torch.Tensor, ratio: Optional[float] = None) -> torch.Tensor:
        """
        Generate random mask for node masking

        Args:
            batch: Batch indices for each node
            ratio: Masking ratio (defaults to cfg.mask_ratio)

        Returns:
            Boolean tensor [num_nodes] indicating masked nodes
        """
        if ratio is None:
            ratio = self.cfg.mask_ratio

        num_nodes = batch.numel()
        num_mask = int(num_nodes * ratio)

        # Randomly select nodes to mask
        perm = torch.randperm(num_nodes, device=batch.device)
        mask_indices = perm[:num_mask]

        mask = torch.zeros(num_nodes, dtype=torch.bool, device=batch.device)
        mask[mask_indices] = True

        return mask

    def predict_properties(self, batch) -> torch.Tensor:
        """Inference: predict molecular properties"""
        with torch.no_grad():
            graph_emb = self.backbone(batch)
            props = self.property_head(graph_emb)
        return props

    def extract_embeddings(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings for downstream tasks"""
        with torch.no_grad():
            graph_emb, node_emb = self.backbone(batch, return_node_embeddings=True)
        return graph_emb, node_emb


class ZINC20ContextOnly(nn.Module):
    """Simplified model with only context prediction task"""
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone = GATBackbone(
            in_dim=cfg.in_dim,
            hidden=cfg.hidden,
            heads=cfg.heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pool=cfg.pool
        )

        self.context_head = ContextPredictionHead(
            hidden_dim=cfg.hidden,
            num_atom_types=cfg.num_atom_types
        )

    def forward(self, batch) -> torch.Tensor:
        graph_emb, node_emb = self.backbone(batch, return_node_embeddings=True)
        context_logits = self.context_head(node_emb)

        if batch.context is not None:
            loss = F.binary_cross_entropy_with_logits(
                context_logits,
                batch.context,
                reduction='mean'
            )
            return loss
        return context_logits


class ZINC20PropertyOnly(nn.Module):
    """Simplified model with only property prediction task"""
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone = GATBackbone(
            in_dim=cfg.in_dim,
            hidden=cfg.hidden,
            heads=cfg.heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pool=cfg.pool
        )

        self.property_head = PropertyPredictionHead(
            hidden_dim=cfg.hidden,
            num_props=cfg.num_props
        )

    def forward(self, batch) -> torch.Tensor:
        graph_emb = self.backbone(batch)
        prop_logits = self.property_head(graph_emb)

        if batch.props is not None:
            loss = F.mse_loss(prop_logits, batch.props, reduction='mean')
            return loss
        return prop_logits


def load_pretrained_backbone(
    checkpoint_path: str | Path,
    cfg: PretrainConfig,
    freeze: bool = False
) -> GATBackbone:
    """
    Load pretrained backbone for fine-tuning

    Args:
        checkpoint_path: Path to pretrained checkpoint
        cfg: Configuration matching the pretrained model
        freeze: If True, freeze backbone weights

    Returns:
        GATBackbone with pretrained weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create backbone
    backbone = GATBackbone(
        in_dim=cfg.in_dim,
        hidden=cfg.hidden,
        heads=cfg.heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pool=cfg.pool
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Extract only backbone weights
        backbone_state = {
            k.replace('backbone.', ''): v
            for k, v in state_dict.items()
            if k.startswith('backbone.')
        }
        backbone.load_state_dict(backbone_state)
    elif 'backbone' in checkpoint:
        backbone.load_state_dict(checkpoint['backbone'])
    else:
        raise ValueError(f"Cannot extract backbone from checkpoint: {checkpoint_path}")

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone


if __name__ == "__main__":
    # Test model creation
    print("Testing ZINC20 Pretraining Model")

    cfg = PretrainConfig(
        in_dim=29,
        hidden=128,
        heads=4,
        num_layers=3,
        dropout=0.2
    )

    model = ZINC20PretrainModel(cfg)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass (dummy data)
    from torch_geometric.data import Batch, Data

    dummy_data = [
        Data(
            x=torch.randn(10, 29),
            edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]]),
            context=torch.rand(10, 9),
            props=torch.randn(9)
        )
    ]
    batch = Batch.from_data_list(dummy_data)

    output = model(batch, return_components=True)
    print(f"Forward pass successful!")
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
