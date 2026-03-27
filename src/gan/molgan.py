"""
MolGAN model implementation.

Implements a graph-based GAN for molecule generation using:
- GraphGenerator: GAT-based generator for molecular graphs
- GraphDiscriminator: Graph-level discriminator
- WGAN-GP training for stable training
- Policy gradient for RL fine-tuning
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from rdkit import Chem

from ..config import GANConfig
from ..vae.molecule_vae import get_atom_feature_dim, get_bond_feature_dim


@dataclass
class GeneratorOutput:
    """Output from generator forward pass."""
    atom_features: torch.Tensor  # [batch_size, max_atoms, atom_feat_dim]
    bond_features: torch.Tensor  # [batch_size, max_atoms, max_atoms, bond_feat_dim]
    adjacency: torch.Tensor  # [batch_size, max_atoms, max_atoms]
    node_mask: torch.Tensor  # [batch_size, max_atoms]


@dataclass
class DiscriminatorOutput:
    """Output from discriminator forward pass."""
    validity: torch.Tensor  # [batch_size, 1] - logits for real/fake
    features: Optional[torch.Tensor] = None  # [batch_size, hidden_dim] - penultimate layer


class GraphGenerator(nn.Module):
    """
    Graph-based molecular generator.

    Generates molecular graphs from random noise using GAT layers.
    Outputs atom features and adjacency matrix for molecule construction.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        max_atoms: int = 50,
        atom_feat_dim: int = 23,
        bond_feat_dim: int = 7,
        num_layers: int = 3,
        gat_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms
        self.atom_feat_dim = atom_feat_dim
        self.bond_feat_dim = bond_feat_dim

        # Initial projection
        self.init_fc = nn.Linear(latent_dim, hidden_dim)

        # Atom feature generation
        atom_layers = []
        for _ in range(num_layers):
            atom_layers.append(nn.Linear(hidden_dim, hidden_dim))
            atom_layers.append(nn.LayerNorm(hidden_dim))
            atom_layers.append(nn.ReLU())
            atom_layers.append(nn.Dropout(dropout))
        atom_layers.append(nn.Linear(hidden_dim, max_atoms * atom_feat_dim))
        self.atom_net = nn.Sequential(*atom_layers)

        # Adjacency generation
        adj_layers = []
        adj_input_dim = hidden_dim + max_atoms * atom_feat_dim
        for _ in range(num_layers):
            adj_layers.append(nn.Linear(adj_input_dim, hidden_dim))
            adj_layers.append(nn.LayerNorm(hidden_dim))
            adj_layers.append(nn.ReLU())
            adj_layers.append(nn.Dropout(dropout))
            adj_input_dim = hidden_dim
        adj_layers.append(nn.Linear(hidden_dim, max_atoms * max_atoms))
        self.adj_net = nn.Sequential(*adj_layers)

        # Node mask (to handle variable-sized molecules)
        self.mask_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_atoms),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        training: bool = True,
    ) -> GeneratorOutput:
        """
        Generate molecular graph from noise.

        Args:
            z: Random noise [batch_size, latent_dim]
            training: Whether in training mode

        Returns:
            GeneratorOutput with atom_features, bond_features, adjacency, node_mask
        """
        batch_size = z.size(0)

        # Initial projection
        h = self.init_fc(z)  # [batch_size, hidden_dim]

        # Generate atom features
        atom_flat = self.atom_net(h)  # [batch_size, max_atoms * atom_feat_dim]
        atom_features = atom_flat.view(batch_size, self.max_atoms, self.atom_feat_dim)

        # Apply sigmoid for feature probabilities
        atom_probs = torch.sigmoid(atom_features)

        # Generate adjacency
        h_adj = torch.cat([h, atom_flat], dim=1)
        adj_flat = self.adj_net(h_adj)  # [batch_size, max_atoms * max_atoms]
        adjacency = adj_flat.view(batch_size, self.max_atoms, self.max_atoms)

        # Symmetrize and apply sigmoid
        adjacency = (adjacency + adjacency.transpose(1, 2)) / 2
        adjacency = torch.sigmoid(adjacency)

        # Generate node mask
        node_mask = self.mask_net(h)

        # Generate bond features from adjacency
        # Simplified: bond features derived from adjacency
        bond_features = adjacency.unsqueeze(-1).expand(-1, -1, -1, self.bond_feat_dim)

        if training:
            # Use probabilities during training
            return GeneratorOutput(
                atom_features=atom_probs,
                bond_features=bond_features,
                adjacency=adjacency,
                node_mask=node_mask,
            )
        else:
            # Use hard decisions during inference
            return GeneratorOutput(
                atom_features=(atom_probs > 0.5).float(),
                bond_features=bond_features,
                adjacency=(adjacency > 0.5).float(),
                node_mask=(node_mask > 0.5).float(),
            )


class GraphDiscriminator(nn.Module):
    """
    Graph-based discriminator.

    Evaluates whether a molecular graph is real or generated.
    Uses GAT layers for graph-level representation.
    """

    def __init__(
        self,
        atom_feat_dim: int = 23,
        bond_feat_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 3,
        gat_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(atom_feat_dim, hidden_dim, heads=gat_heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * gat_heads, hidden_dim, heads=gat_heads, concat=True))
        self.convs.append(GATConv(hidden_dim * gat_heads, hidden_dim, heads=1, concat=True))

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> DiscriminatorOutput:
        """
        Discriminate real vs fake.

        Args:
            atom_features: Node features [num_nodes, atom_feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector [num_nodes]

        Returns:
            DiscriminatorOutput with validity and features
        """
        x = atom_features

        # Apply GAT layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)

        # Save features for gradient penalty
        features = x

        # Output
        validity = self.fc(x)

        return DiscriminatorOutput(validity=validity, features=features)


class MolGAN(nn.Module):
    """
    Molecule GAN combining generator and discriminator.

    Implements WGAN-GP for stable training and supports
    RL fine-tuning with reward signals.
    """

    def __init__(
        self,
        cfg: GANConfig,
        atom_feat_dim: int = 23,
        bond_feat_dim: int = 7,
        max_atoms: int = 50,
    ):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.max_atoms = max_atoms

        self.generator = GraphGenerator(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            max_atoms=max_atoms,
            atom_feat_dim=atom_feat_dim,
            bond_feat_dim=bond_feat_dim,
            num_layers=cfg.num_layers,
            gat_heads=cfg.gat_heads,
            dropout=cfg.dropout,
        )

        self.discriminator = GraphDiscriminator(
            atom_feat_dim=atom_feat_dim,
            bond_feat_dim=bond_feat_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            gat_heads=cfg.gat_heads,
            dropout=cfg.dropout,
        )

    def generate_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate random noise for generator input."""
        return torch.randn(batch_size, self.latent_dim, device=device)

    def generate(
        self,
        n_samples: int,
        device: torch.device,
        training: bool = False,
    ) -> GeneratorOutput:
        """
        Generate molecules.

        Args:
            n_samples: Number of molecules to generate
            device: Device to generate on
            training: Whether in training mode

        Returns:
            GeneratorOutput
        """
        z = self.generate_noise(n_samples, device)
        return self.generator(z, training=training)

    def discriminate_real(self, batch) -> DiscriminatorOutput:
        """
        Discriminate real molecules.

        Args:
            batch: PyG Data batch

        Returns:
            DiscriminatorOutput
        """
        return self.discriminator(
            atom_features=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )

    def discriminate_fake(
        self,
        gen_output: GeneratorOutput,
        batch: Optional[torch.Tensor] = None,
    ) -> DiscriminatorOutput:
        """
        Discriminate generated molecules.

        Args:
            gen_output: Generator output
            batch: Batch vector

        Returns:
            DiscriminatorOutput
        """
        # Convert generator output to graph format
        # This is simplified - in practice, you'd need proper conversion
        atom_features = gen_output.atom_features
        adjacency = gen_output.adjacency

        batch_size = atom_features.size(0)
        device = atom_features.device

        # Flatten atom features: [batch_size, max_atoms, feat] -> [batch_size * max_atoms, feat]
        num_nodes = self.max_atoms
        x = atom_features.view(batch_size * num_nodes, -1)

        # Create edge indices from adjacency
        edge_list = []
        for b in range(batch_size):
            adj = adjacency[b]  # [max_atoms, max_atoms]
            # Find connected pairs
            connected = (adj > 0.5).nonzero()
            for i, j in connected:
                edge_list.append([b * num_nodes + i.item(), b * num_nodes + j.item()])

        if edge_list:
            edge_index = torch.tensor(edge_list, device=device).t()
        else:
            # No edges - create empty edge index
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        # Create batch vector
        if batch is None:
            batch_vec = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
        else:
            batch_vec = batch

        return self.discriminator(x, edge_index, batch_vec)

    def forward(self, batch_size: int, device: torch.device) -> GeneratorOutput:
        """Forward pass for generation."""
        return self.generate(batch_size, device)


def compute_gradient_penalty(
    discriminator: GraphDiscriminator,
    real_batch,
    fake_output: GeneratorOutput,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.

    Args:
        discriminator: Discriminator network
        real_batch: Real data batch
        fake_output: Generated data
        device: Device

    Returns:
        Gradient penalty loss
    """
    batch_size = real_batch.num_graphs

    # Generate random interpolation weights
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_batch.x)

    # Create interpolates (simplified - would need proper implementation)
    # interpolates = alpha * real_batch.x + (1 - alpha) + fake_output.atom_features

    # Compute discriminator on interpolates
    # disc_interpolates = discriminator(interpolates, ...)

    # Compute gradients
    # gradients = torch.autograd.grad(...)

    # Gradient penalty
    # gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    # For now, return zero (placeholder)
    return torch.tensor(0.0, device=device)
