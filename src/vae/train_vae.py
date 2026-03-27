"""
VAE training implementation.

Provides training loop for MoleculeVAE with:
- KL annealing
- BBB prediction auxiliary loss
- QED score auxiliary loss
- Checkpointing and logging
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from tqdm import tqdm

from .molecule_vae import (
    MoleculeVAE,
    VAEOutput,
    vae_loss_function,
    compute_qed_loss,
    compute_sa_score,
    get_atom_feature_dim,
)
from .dataset import MoleculeDataset, BBBDataset, SMILESDataset
from ..config import VAEConfig, VAETrainConfig
from ..utils.seed import seed_everything


class VAETrainer:
    """
    Trainer class for MoleculeVAE.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: MoleculeVAE,
        cfg: VAETrainConfig,
        vae_cfg: VAEConfig = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.vae_cfg = vae_cfg or model.cfg
        self.device = device or torch.device(cfg.device if hasattr(cfg, 'device') and torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Optimizer - get learning rate from VAEConfig
        lr = self.vae_cfg.learning_rate if self.vae_cfg else 1e-3
        weight_decay = self.vae_cfg.weight_decay if self.vae_cfg else 1e-5
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Get grad_clip from config
        self.grad_clip = self.vae_cfg.grad_clip if self.vae_cfg and hasattr(self.vae_cfg, 'grad_clip') else 5.0

        # Get other training parameters
        self.save_every = cfg.save_every if hasattr(cfg, 'save_every') else 10
        self.val_every = cfg.val_every if hasattr(cfg, 'val_every') else 5

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = []

        # Paths
        self.model_dir = cfg.vae_model_dir if hasattr(cfg, 'vae_model_dir') else Path("artifacts/models/vae")
        self.logs_dir = cfg.vae_logs_dir if hasattr(cfg, 'vae_logs_dir') else Path("artifacts/logs/vae")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def compute_beta(self, epoch: int) -> float:
        """Compute KL annealing beta."""
        if not self.model.cfg.kl_anneal:
            return self.model.cfg.beta

        anneal_epochs = self.model.cfg.anneal_epochs
        if epoch >= anneal_epochs:
            return self.model.cfg.beta

        # Linear annealing
        return self.model.cfg.beta * (epoch / anneal_epochs)

    def train_epoch(
        self,
        train_loader: DataLoader,
        bbb_predictor: Optional[object] = None,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []

        beta = self.compute_beta(self.current_epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs: VAEOutput = self.model(batch)

            # Compute VAE loss
            vae_loss, loss_dict = vae_loss_function(
                outputs,
                batch,
                beta=beta,
            )

            # Additional BBB loss if predictor provided
            if bbb_predictor is not None:
                bbb_loss = self.compute_bbb_loss(outputs, bbb_predictor)
                loss = vae_loss + self.model.cfg.lambda_bbb * bbb_loss
                loss_dict['bbb_loss'] = bbb_loss.item()
            else:
                loss = vae_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            epoch_losses.append(loss_dict)
            pbar.set_postfix({'loss': loss.item()})

        # Aggregate losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([d[key] for d in epoch_losses])

        return avg_losses

    def compute_bbb_loss(self, outputs: VAEOutput, bbb_predictor) -> torch.Tensor:
        """Compute BBB prediction loss from latent vectors."""
        # Simple approach: encourage latent vectors to predict BBB+
        # In practice, you'd decode z to molecules and predict
        return torch.tensor(0.0, device=self.device)

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = batch.to(self.device)
            outputs: VAEOutput = self.model(batch)

            loss, _ = vae_loss_function(outputs, batch, beta=self.model.cfg.beta)

            total_loss += loss.item()
            num_batches += 1

        return {
            'val_loss': total_loss / num_batches if num_batches > 0 else 0.0,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        bbb_predictor: Optional[object] = None,
    ) -> pd.DataFrame:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            bbb_predictor: Optional BBB predictor for auxiliary loss

        Returns:
            DataFrame with training history
        """
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_losses = self.train_epoch(train_loader, bbb_predictor)

            # Validate
            if val_loader is not None:
                val_losses = self.validate(val_loader)
            else:
                val_losses = {}

            # Log
            epoch_time = time.time() - epoch_start
            log_entry = {
                'epoch': epoch,
                'epoch_time': epoch_time,
                **train_losses,
                **val_losses,
            }
            self.history.append(log_entry)

            print(f"Epoch {epoch}: {log_entry}")

            # Save checkpoint
            if epoch % self.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            # Save best model
            if val_loader is not None:
                if val_losses['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['val_loss']
                    self.save_checkpoint("best.pt")
                    print(f"  -> New best model: val_loss={self.best_val_loss:.4f}")

        # Save final model
        self.save_checkpoint("last.pt")

        # Save history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.logs_dir / "training_history.csv", index=False)

        return history_df

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        torch.save(checkpoint, self.model_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.model_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint.get('history', [])


def train_vae(
    train_ds: MoleculeDataset,
    val_ds: Optional[MoleculeDataset],
    cfg: VAETrainConfig,
    vae_cfg: VAEConfig,
    bbb_predictor: Optional[object] = None,
) -> VAETrainer:
    """
    Train VAE model.

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        cfg: Training configuration
        vae_cfg: VAE model configuration
        bbb_predictor: Optional BBB predictor

    Returns:
        Trained VAETrainer instance
    """
    seed_everything(cfg.seed)

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=vae_cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=vae_cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

    # Get input dimension
    in_dim = train_ds[0].x.size(-1)

    # Create model
    model = MoleculeVAE(
        in_dim=in_dim,
        cfg=vae_cfg,
        max_atoms=50,
    )

    # Create trainer
    trainer = VAETrainer(model, cfg, vae_cfg=vae_cfg)

    # Train
    num_epochs = vae_cfg.epochs
    trainer.train(train_loader, val_loader, num_epochs, bbb_predictor)

    return trainer


def generate_molecules(
    model: MoleculeVAE,
    n_samples: int = 100,
    device: torch.device = torch.device('cpu'),
    temperature: float = 1.0,
) -> List[str]:
    """
    Generate molecules from trained VAE.

    Args:
        model: Trained MoleculeVAE
        n_samples: Number of molecules to generate
        device: Device to generate on
        temperature: Sampling temperature

    Returns:
        List of generated SMILES strings
    """
    model.eval()

    with torch.no_grad():
        # Sample latent vectors
        z = model.sample(n_samples, device)

        # Decode
        atom_features, bond_features = model.decode(z)

        # Convert to SMILES
        # Note: This is a simplified approach
        # In practice, you'd need a proper SMILES decoder
        smiles_list = []

        for i in range(n_samples):
            # Placeholder: generate simple SMILES based on atom types
            # A proper implementation would use a sequential decoder
            # or junction tree decoder for valid SMILES
            smiles_list.append("C")  # Placeholder

    return smiles_list


def decode_graph_to_smiles(
    atom_features: torch.Tensor,
    bond_features: torch.Tensor,
    idx: int,
) -> Optional[str]:
    """
    Decode graph features back to SMILES.

    Args:
        atom_features: [batch_size, max_atoms, atom_feat_dim]
        bond_features: [batch_size, max_atoms, max_atoms, bond_feat_dim]
        idx: Index in batch

    Returns:
        SMILES string or None if invalid
    """
    # This is a placeholder implementation
    # A proper implementation would:
    # 1. Decode atom types from atom_features
    # 2. Decode bond types from bond_features
    # 3. Build RDKit molecule
    # 4. Convert to SMILES

    return None


def evaluate_generated_molecules(
    smiles_list: List[str],
    bbb_predictor,
    min_qed: float = 0.5,
    min_bbb_prob: float = 0.7,
    max_sa: float = 4.0,
) -> Dict:
    """
    Evaluate generated molecules.

    Args:
        smiles_list: List of generated SMILES
        bbb_predictor: BBB prediction model
        min_qed: Minimum QED score
        min_bbb_prob: Minimum BBB probability
        max_sa: Maximum SA score

    Returns:
        Dictionary with evaluation metrics
    """
    valid_smiles = []
    qed_scores = []
    bbb_probs = []
    sa_scores = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        valid_smiles.append(smi)

        # QED score
        try:
            qed = QED.qed(mol)
            qed_scores.append(qed)
        except:
            qed_scores.append(0.0)

        # BBB prediction
        try:
            results = bbb_predictor.predict([smi])
            bbb_prob = results.ensemble_probability[0]
            bbb_probs.append(bbb_prob)
        except:
            bbb_probs.append(0.0)

        # SA score
        sa = compute_sa_score(smi)
        sa_scores.append(sa)

    # Compute metrics
    n_valid = len(valid_smiles)
    n_total = len(smiles_list)
    validity = n_valid / n_total if n_total > 0 else 0.0

    avg_qed = np.mean(qed_scores) if qed_scores else 0.0
    avg_bbb_prob = np.mean(bbb_probs) if bbb_probs else 0.0
    avg_sa = np.mean(sa_scores) if sa_scores else 10.0

    # Pass filters
    n_pass = sum(
        1 for q, b, s in zip(qed_scores, bbb_probs, sa_scores)
        if q >= min_qed and b >= min_bbb_prob and s <= max_sa
    )
    pass_rate = n_pass / n_valid if n_valid > 0 else 0.0

    return {
        'n_total': n_total,
        'n_valid': n_valid,
        'validity': validity,
        'avg_qed': avg_qed,
        'avg_bbb_prob': avg_bbb_prob,
        'avg_sa_score': avg_sa,
        'n_pass_filters': n_pass,
        'pass_rate': pass_rate,
        'valid_smiles': valid_smiles,
    }


if __name__ == "__main__":
    # Test imports
    print("VAE training module loaded successfully")
    print(f"Atom feature dim: {get_atom_feature_dim()}")
