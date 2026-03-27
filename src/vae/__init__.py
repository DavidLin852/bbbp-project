"""
VAE (Variational Autoencoder) module for molecule generation.

This module implements a graph-based VAE for generating BBB-permeable molecules.
The VAE learns a latent representation of molecular structures and can generate
new molecules by sampling from the latent space.

Key components:
- MoleculeVAE: Main VAE model with encoder and decoder
- GraphEncoder: GAT-based encoder for molecular graphs
- GraphDecoder: Decoder for reconstructing molecular graphs
- train_vae: Training script for the VAE
"""

from .molecule_vae import MoleculeVAE, GraphEncoder, GraphDecoder, VAEOutput
from .dataset import MoleculeDataset, BBBDataset
from .train_vae import VAETrainer, train_vae, generate_molecules

__all__ = [
    'MoleculeVAE',
    'GraphEncoder',
    'GraphDecoder',
    'VAEOutput',
    'MoleculeDataset',
    'BBBDataset',
    'VAETrainer',
    'train_vae',
    'generate_molecules',
]
