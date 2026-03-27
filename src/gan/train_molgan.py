"""
MolGAN training implementation.

Provides training loop for MolGAN with:
- WGAN-GP for stable adversarial training
- RL fine-tuning with multi-objective rewards
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
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .molgan import MolGAN, GeneratorOutput, compute_gradient_penalty
from .reward import RewardNetwork, compute_rewards, policy_gradient_loss, BaselineNetwork
from ..config import GANConfig, GANTrainConfig
from ..utils.seed import seed_everything


class MolGANTrainer:
    """
    Trainer class for MolGAN.

    Handles alternating G/D training, RL fine-tuning, and logging.
    """

    def __init__(
        self,
        model: MolGAN,
        cfg: GANTrainConfig,
        gan_cfg: GANConfig,
        bbb_predictor,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.gan_cfg = gan_cfg
        self.bbb_predictor = bbb_predictor
        self.device = device or torch.device(cfg.device if hasattr(cfg, 'device') and torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Optimizers - get learning rates from gan_cfg
        g_lr = gan_cfg.g_lr if hasattr(gan_cfg, 'g_lr') else 1e-4
        d_lr = gan_cfg.d_lr if hasattr(gan_cfg, 'd_lr') else 1e-4

        self.g_optimizer = optim.Adam(
            self.model.generator.parameters(),
            lr=g_lr,
            betas=(0.5, 0.999),
        )
        self.d_optimizer = optim.Adam(
            self.model.discriminator.parameters(),
            lr=d_lr,
            betas=(0.5, 0.999),
        )

        # Reward network and baseline
        self.reward_net = RewardNetwork(
            bbb_predictor=bbb_predictor,
            bbb_weight=gan_cfg.reward_bbb,
            qed_weight=gan_cfg.reward_qed,
            sa_weight=gan_cfg.reward_sa,
            validity_weight=gan_cfg.reward_validity,
            min_qed=gan_cfg.min_qed,
            min_bbb_prob=gan_cfg.min_bbb_prob,
            max_sa=gan_cfg.max_sa_score,
        )

        self.baseline = BaselineNetwork(
            latent_dim=gan_cfg.latent_dim,
            hidden_dim=gan_cfg.hidden_dim,
        ).to(self.device)

        self.baseline_optimizer = optim.Adam(
            self.baseline.parameters(),
            lr=g_lr,
        )

        # Get other config parameters
        self.grad_clip = gan_cfg.grad_clip if hasattr(gan_cfg, 'grad_clip') else 5.0
        self.save_every = cfg.save_every if hasattr(cfg, 'save_every') else 10
        self.val_every = cfg.val_every if hasattr(cfg, 'val_every') else 5

        # Training state
        self.current_epoch = 0
        self.best_score = -float('inf')
        self.history = []

        # Paths
        self.model_dir = cfg.gan_model_dir if hasattr(cfg, 'gan_model_dir') else Path("artifacts/models/gan")
        self.logs_dir = cfg.gan_logs_dir if hasattr(cfg, 'gan_logs_dir') else Path("artifacts/logs/gan")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def train_discriminator(
        self,
        real_loader: DataLoader,
        n_critic: int = 5,
    ) -> Dict[str, float]:
        """Train discriminator for n_critic iterations."""
        self.model.discriminator.train()
        self.model.generator.eval()

        d_losses = []

        for _ in range(n_critic):
            # Get real batch
            real_batch = next(iter(real_loader)).to(self.device)

            # Generate fake batch
            fake_output = self.model.generate(
                real_batch.num_graphs,
                self.device,
                training=True,
            )

            # Discriminator real
            real_output = self.model.discriminate_real(real_batch)
            real_loss = -real_output.validity.mean()

            # Discriminator fake
            fake_output_disc = self.model.discriminate_fake(fake_output)
            fake_loss = fake_output_disc.validity.mean()

            # Gradient penalty
            gp = compute_gradient_penalty(
                self.model.discriminator,
                real_batch,
                fake_output,
                self.device,
            )

            # Total loss
            d_loss = real_loss + fake_loss + self.gan_cfg.gp_weight * gp

            # Optimize
            self.d_optimizer.zero_grad()
            d_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.discriminator.parameters(),
                    self.grad_clip,
                )
            self.d_optimizer.step()

            d_losses.append({
                'd_loss': d_loss.item(),
                'real_loss': real_loss.item(),
                'fake_loss': fake_loss.item(),
                'gp': gp.item(),
            })

        # Aggregate
        avg_losses = {}
        for key in d_losses[0].keys():
            avg_losses[key] = np.mean([d[key] for d in d_losses])

        return avg_losses

    def train_generator(
        self,
        real_loader: DataLoader,
        use_rl: bool = False,
    ) -> Dict[str, float]:
        """Train generator."""
        self.model.generator.train()
        self.model.discriminator.eval()

        # Get batch size
        real_batch = next(iter(real_loader)).to(self.device)
        batch_size = real_batch.num_graphs

        # Generate
        z = self.model.generate_noise(batch_size, self.device)
        fake_output = self.model.generator(z, training=True)

        # Discriminator loss
        fake_output_disc = self.model.discriminate_fake(fake_output)
        g_loss_adv = -fake_output_disc.validity.mean()

        if use_rl:
            # RL fine-tuning
            # Decode to SMILES and compute rewards
            smiles_list = self.decode_output_to_smiles(fake_output)

            reward_output = self.reward_net(smiles_list)
            rewards = reward_output.total_reward

            # Baseline prediction
            baseline_pred = self.baseline(z)

            # Policy gradient loss
            # Simplified - in practice, you'd need proper log probs
            g_loss_rl = -(rewards - baseline_pred.detach()).mean()

            # Update baseline
            baseline_loss = F.mse_loss(baseline_pred.squeeze(), rewards)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

            # Combined loss
            g_loss = g_loss_adv + self.gan_cfg.rl_weight * g_loss_rl

            loss_dict = {
                'g_loss': g_loss.item(),
                'g_loss_adv': g_loss_adv.item(),
                'g_loss_rl': g_loss_rl.item(),
                'baseline_loss': baseline_loss.item(),
                'avg_reward': rewards.mean().item(),
                'validity': np.mean(reward_output.metrics['valid']),
                'avg_qed': np.mean(reward_output.metrics['qed_scores']) if reward_output.metrics['qed_scores'] else 0.0,
                'avg_bbb': np.mean(reward_output.metrics['bbb_probs']) if reward_output.metrics['bbb_probs'] else 0.0,
            }
        else:
            g_loss = g_loss_adv
            loss_dict = {
                'g_loss': g_loss.item(),
                'g_loss_adv': g_loss_adv.item(),
            }

        # Optimize
        self.g_optimizer.zero_grad()
        g_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.generator.parameters(),
                self.grad_clip,
            )
        self.g_optimizer.step()

        return loss_dict

    def decode_output_to_smiles(self, output: GeneratorOutput) -> List[str]:
        """
        Decode generator output to SMILES strings.

        Args:
            output: Generator output

        Returns:
            List of SMILES strings
        """
        # Placeholder implementation
        # In practice, this would use a proper decoder
        batch_size = output.atom_features.size(0)
        return ["C"] * batch_size

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 200,
    ) -> pd.DataFrame:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train

        Returns:
            DataFrame with training history
        """
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Check if RL phase
            use_rl = epoch >= self.gan_cfg.rl_start_epoch

            phase = "RL" if use_rl else "Warmup"

            # Train discriminator
            d_losses = self.train_discriminator(
                train_loader,
                n_critic=self.gan_cfg.n_critic,
            )

            # Train generator
            g_losses = self.train_generator(train_loader, use_rl=use_rl)

            # Log
            epoch_time = time.time() - epoch_start
            log_entry = {
                'epoch': epoch,
                'phase': phase,
                'epoch_time': epoch_time,
                **d_losses,
                **g_losses,
            }
            self.history.append(log_entry)

            print(f"Epoch {epoch} ({phase}): {log_entry}")

            # Save checkpoint
            if epoch % self.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            # Evaluate and save best
            if val_loader is not None and epoch % self.val_every == 0:
                metrics = self.evaluate(val_loader)
                score = metrics.get('score', 0.0)

                if score > self.best_score:
                    self.best_score = score
                    self.save_checkpoint("best.pt")
                    print(f"  -> New best model: score={score:.4f}")

        # Save final
        self.save_checkpoint("last.pt")

        # Save history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.logs_dir / "training_history.csv", index=False)

        return history_df

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate generator."""
        self.model.generator.eval()

        # Generate molecules
        batch = next(iter(val_loader))
        n_samples = batch.num_graphs

        fake_output = self.model.generate(
            n_samples,
            self.device,
            training=False,
        )

        # Decode to SMILES
        smiles_list = self.decode_output_to_smiles(fake_output)

        # Compute rewards
        reward_output = self.reward_net(smiles_list)

        return {
            'score': reward_output.total_reward.mean().item(),
            'validity': np.mean(reward_output.metrics['valid']),
            'avg_qed': np.mean(reward_output.metrics['qed_scores']) if reward_output.metrics['qed_scores'] else 0.0,
            'avg_bbb': np.mean(reward_output.metrics['bbb_probs']) if reward_output.metrics['bbb_probs'] else 0.0,
        }

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'generator': self.model.generator.state_dict(),
            'discriminator': self.model.discriminator.state_dict(),
            'baseline': self.baseline.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'best_score': self.best_score,
            'history': self.history,
        }
        torch.save(checkpoint, self.model_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load checkpoint."""
        checkpoint = torch.load(self.model_dir / filename, map_location=self.device)
        self.model.generator.load_state_dict(checkpoint['generator'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator'])
        self.baseline.load_state_dict(checkpoint['baseline'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.current_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        self.history = checkpoint.get('history', [])


def train_gan(
    train_ds,
    val_ds,
    cfg: GANTrainConfig,
    gan_cfg: GANConfig,
    bbb_predictor,
) -> MolGANTrainer:
    """
    Train MolGAN model.

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        cfg: Training configuration
        gan_cfg: GAN model configuration
        bbb_predictor: BBB prediction model

    Returns:
        Trained MolGANTrainer instance
    """
    seed_everything(cfg.seed)

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=gan_cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=gan_cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

    # Create model
    model = MolGAN(cfg=gan_cfg)

    # Create trainer
    trainer = MolGANTrainer(model, cfg, gan_cfg, bbb_predictor)

    # Train
    num_epochs = gan_cfg.epochs
    trainer.train(train_loader, val_loader, num_epochs)

    return trainer


def generate_with_gan(
    model: MolGAN,
    n_samples: int = 100,
    device: torch.device = torch.device('cpu'),
) -> List[str]:
    """
    Generate molecules using trained GAN.

    Args:
        model: Trained MolGAN
        n_samples: Number of molecules to generate
        device: Device to generate on

    Returns:
        List of generated SMILES strings
    """
    model.generator.eval()

    with torch.no_grad():
        output = model.generate(
            n_samples,
            device,
            training=False,
        )

        # Decode to SMILES
        # Placeholder implementation
        smiles_list = ["C"] * n_samples

    return smiles_list
