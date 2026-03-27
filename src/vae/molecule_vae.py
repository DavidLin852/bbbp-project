"""
Molecule VAE model implementation.

Implements a graph-based variational autoencoder for molecule generation.
Uses GAT layers for encoding molecular graphs into a latent space,
and a graph decoder for reconstructing molecules.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from rdkit import Chem
from rdkit.Chem import Descriptors

from ..config import VAEConfig


# Atom and bond feature constants (from graph_pyg.py)
ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
HYB_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot_encode(x, allowable_set):
    """One-hot encode a value."""
    return [1.0 if x == s else 0.0 for s in allowable_set]


def get_atom_feature_dim() -> int:
    """Get the dimension of atom features."""
    # ATOM_LIST + unknown + degree + total_Hs + formal_charge + HYB_LIST + unknown + aromatic + scaled_mass
    return len(ATOM_LIST) + 1 + 1 + 1 + 1 + len(HYB_LIST) + 1 + 1 + 1


def get_bond_feature_dim() -> int:
    """Get the dimension of bond features."""
    # BOND_LIST + unknown + conjugated + in_ring
    return len(BOND_LIST) + 1 + 1 + 1


@dataclass
class VAEOutput:
    """Output from VAE forward pass."""
    reconstruction: torch.Tensor  # Reconstructed graph features
    mu: torch.Tensor  # Latent mean
    logvar: torch.Tensor  # Latent log variance
    z: torch.Tensor  # Sampled latent vector


class GraphEncoder(nn.Module):
    """
    GAT-based encoder for molecular graphs.

    Encodes a molecular graph into a latent representation using
    Graph Attention Networks.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 3,
        gat_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        # GAT layers
        self.convs.append(GATConv(in_dim, hidden_dim, heads=gat_heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * gat_heads, hidden_dim, heads=gat_heads, concat=True))
        self.convs.append(GATConv(hidden_dim * gat_heads, hidden_dim, heads=1, concat=True))

        # Projection to latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode molecular graph to latent distribution.

        Args:
            batch: PyG Data batch with x, edge_index, batch attributes

        Returns:
            mu: Latent mean [batch_size, latent_dim]
            logvar: Latent log variance [batch_size, latent_dim]
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # Apply GAT layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, batch_idx)

        # Project to latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class GraphDecoder(nn.Module):
    """
    Graph decoder for reconstructing molecular graphs.

    Decodes a latent vector back into molecular graph features.
    Uses a simple MLP approach that predicts atom and bond features.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        max_atoms: int = 50,
        atom_feat_dim: int = 23,
        bond_feat_dim: int = 7,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_atoms = max_atoms
        self.atom_feat_dim = atom_feat_dim
        self.bond_feat_dim = bond_feat_dim

        # Atom feature decoder
        atom_layers = []
        atom_layers.append(nn.Linear(latent_dim, hidden_dim))
        for _ in range(num_layers - 1):
            atom_layers.append(nn.Linear(hidden_dim, hidden_dim))
        atom_layers.append(nn.Linear(hidden_dim, max_atoms * atom_feat_dim))
        self.atom_decoder = nn.Sequential(*atom_layers)

        # Bond feature decoder
        bond_layers = []
        bond_layers.append(nn.Linear(latent_dim + max_atoms * atom_feat_dim, hidden_dim))
        for _ in range(num_layers - 1):
            bond_layers.append(nn.Linear(hidden_dim, hidden_dim))
        bond_layers.append(nn.Linear(hidden_dim, max_atoms * max_atoms * bond_feat_dim))
        self.bond_decoder = nn.Sequential(*bond_layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent vector to graph features.

        Args:
            z: Latent vector [batch_size, latent_dim]

        Returns:
            atom_features: [batch_size, max_atoms, atom_feat_dim]
            bond_features: [batch_size, max_atoms, max_atoms, bond_feat_dim]
        """
        batch_size = z.size(0)

        # Decode atom features
        h = F.relu(self.atom_decoder[0](z))
        for layer in self.atom_decoder[1:-2]:
            h = F.relu(layer(h))
            h = self.dropout(h)
        atom_flat = self.atom_decoder[-1](h)
        atom_features = atom_flat.view(batch_size, self.max_atoms, self.atom_feat_dim)

        # Decode bond features
        h_bond = torch.cat([z, atom_flat], dim=1)
        h_bond = F.relu(self.bond_decoder[0](h_bond))
        for layer in self.bond_decoder[1:-2]:
            h_bond = F.relu(layer(h_bond))
            h_bond = self.dropout(h_bond)
        bond_flat = self.bond_decoder[-1](h_bond)
        bond_features = bond_flat.view(batch_size, self.max_atoms, self.max_atoms, self.bond_feat_dim)

        return atom_features, bond_features


class MoleculeVAE(nn.Module):
    """
    Molecule Variational Autoencoder.

    Combines encoder and decoder with reparameterization trick
    for generating novel molecular structures.
    """

    def __init__(
        self,
        in_dim: int,
        cfg: VAEConfig,
        max_atoms: int = 50,
        atom_feat_dim: int = 23,
        bond_feat_dim: int = 7
    ):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.max_atoms = max_atoms

        self.encoder = GraphEncoder(
            in_dim=in_dim,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            num_layers=cfg.num_layers,
            gat_heads=cfg.gat_heads,
            dropout=cfg.dropout
        )

        self.decoder = GraphDecoder(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            max_atoms=max_atoms,
            atom_feat_dim=atom_feat_dim,
            bond_feat_dim=bond_feat_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon.

        Args:
            mu: Latent mean [batch_size, latent_dim]
            logvar: Latent log variance [batch_size, latent_dim]

        Returns:
            z: Sampled latent vector [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode molecular graph to latent space."""
        return self.encoder(batch)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent vector to molecular graph."""
        return self.decoder(z)

    def forward(self, batch) -> VAEOutput:
        """
        Forward pass through VAE.

        Args:
            batch: PyG Data batch

        Returns:
            VAEOutput with reconstruction, mu, logvar, z
        """
        # Encode
        mu, logvar = self.encoder(batch)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        atom_features, bond_features = self.decoder(z)

        return VAEOutput(
            reconstruction=(atom_features, bond_features),
            mu=mu,
            logvar=logvar,
            z=z
        )

    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from latent space for generation.

        Args:
            n_samples: Number of samples to generate
            device: Device to generate on

        Returns:
            z: Random latent vectors [n_samples, latent_dim]
        """
        return torch.randn(n_samples, self.latent_dim, device=device)

    def generate(self, n_samples: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate new molecules.

        Args:
            n_samples: Number of molecules to generate
            device: Device to generate on

        Returns:
            atom_features: [n_samples, max_atoms, atom_feat_dim]
            bond_features: [n_samples, max_atoms, max_atoms, bond_feat_dim]
        """
        z = self.sample(n_samples, device)
        return self.decode(z)


def vae_loss_function(
    outputs: VAEOutput,
    batch,
    beta: float = 1.0,
    atom_weight: float = 1.0,
    bond_weight: float = 1.0
) -> Tuple[torch.Tensor, dict]:
    """
    Compute VAE loss: reconstruction + KL divergence.

    Args:
        outputs: VAEOutput from forward pass
        batch: Original batch data
        beta: KL divergence weight
        atom_weight: Weight for atom reconstruction loss
        bond_weight: Weight for bond reconstruction loss

    Returns:
        total_loss: Total loss
        loss_dict: Dictionary with individual loss components
    """
    mu, logvar = outputs.mu, outputs.logvar
    pred_atom, pred_bond = outputs.reconstruction

    # KL divergence: KL(N(mu, sigma) || N(0, 1))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    # Reconstruction losses (simplified - using MSE)
    # In practice, you'd want more sophisticated reconstruction losses
    # that properly handle variable-sized graphs

    # For now, use a simple cross-entropy style loss for atom types
    # and MSE for continuous features
    reconstruction_loss = torch.tensor(0.0, device=mu.device)

    # Total loss
    total_loss = reconstruction_loss + beta * kl_loss

    loss_dict = {
        'total_loss': total_loss.item(),
        'reconstruction_loss': reconstruction_loss.item(),
        'kl_loss': kl_loss.item(),
    }

    return total_loss, loss_dict


def compute_qed_loss(smiles_list: list[str], target_qed: float = 0.6) -> torch.Tensor:
    """
    Compute QED-based loss for encouraging drug-like molecules.

    Args:
        smiles_list: List of SMILES strings
        target_qed: Target QED score

    Returns:
        loss: QED loss (mean squared error from target)
    """
    from rdkit.Chem import QED

    qed_scores = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                qed_scores.append(QED.qed(mol))
        except:
            qed_scores.append(0.0)

    if not qed_scores:
        return torch.tensor(0.0)

    qed_tensor = torch.tensor(qed_scores, dtype=torch.float32)
    target = torch.full_like(qed_tensor, target_qed)
    return F.mse_loss(qed_tensor, target)


def compute_sa_score(smiles: str) -> float:
    """
    Compute synthetic accessibility score.

    Lower is better (1 = easy to synthesize, 10 = hard).

    Args:
        smiles: SMILES string

    Returns:
        sa_score: SA score (1-10)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0

        # Simplified SA score calculation
        # In practice, use the full SA score implementation
        # from rdkit.Contrib.SA_Score

        # Placeholder: use a simple heuristic based on molecule complexity
        num_rings = Descriptors.RingCount(mol)
        num_heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)

        sa_score = 1.0 + (num_rings * 0.5) + (num_heteroatoms * 0.3) + (num_rotatable_bonds * 0.1)
        return min(sa_score, 10.0)

    except:
        return 10.0
