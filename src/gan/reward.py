"""
Reward network for MolGAN RL fine-tuning.

Implements multi-objective reward computation combining:
- BBB permeability prediction
- QED score (drug-likeness)
- SA score (synthetic accessibility)
- Validity (SMILES validity)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

from ..vae.molecule_vae import compute_sa_score


@dataclass
class RewardOutput:
    """Output from reward computation."""
    total_reward: torch.Tensor  # [batch_size] - total reward
    bbb_reward: torch.Tensor  # [batch_size] - BBB permeability reward
    qed_reward: torch.Tensor  # [batch_size] - QED reward
    sa_reward: torch.Tensor  # [batch_size] - SA reward
    validity_reward: torch.Tensor  # [batch_size] - validity reward
    metrics: Dict[str, List[float]]  # Detailed metrics


class RewardNetwork(nn.Module):
    """
    Reward computation network for GAN RL fine-tuning.

    Combines multiple reward signals into a single scalar reward
    for policy gradient optimization.
    """

    def __init__(
        self,
        bbb_predictor,
        bbb_weight: float = 1.0,
        qed_weight: float = 0.3,
        sa_weight: float = 0.3,
        validity_weight: float = 1.0,
        min_qed: float = 0.5,
        min_bbb_prob: float = 0.7,
        max_sa: float = 4.0,
    ):
        super().__init__()
        self.bbb_predictor = bbb_predictor
        self.bbb_weight = bbb_weight
        self.qed_weight = qed_weight
        self.sa_weight = sa_weight
        self.validity_weight = validity_weight
        self.min_qed = min_qed
        self.min_bbb_prob = min_bbb_prob
        self.max_sa = max_sa

    def forward(
        self,
        smiles_list: List[str],
    ) -> RewardOutput:
        """
        Compute rewards for a list of SMILES.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            RewardOutput with rewards and metrics
        """
        batch_size = len(smiles_list)
        device = next(self.bbb_predictor.parameters()).device if hasattr(self.bbb_predictor, 'parameters') else torch.device('cpu')

        # Initialize rewards
        bbb_rewards = torch.zeros(batch_size, device=device)
        qed_rewards = torch.zeros(batch_size, device=device)
        sa_rewards = torch.zeros(batch_size, device=device)
        validity_rewards = torch.zeros(batch_size, device=device)

        # Metrics tracking
        metrics = {
            'valid': [],
            'qed_scores': [],
            'bbb_probs': [],
            'sa_scores': [],
            'valid_smiles': [],
        }

        for i, smi in enumerate(smiles_list):
            # Check validity
            mol = Chem.MolFromSmiles(smi)
            is_valid = mol is not None
            metrics['valid'].append(float(is_valid))

            if not is_valid:
                validity_rewards[i] = 0.0
                continue

            validity_rewards[i] = 1.0
            metrics['valid_smiles'].append(smi)

            # QED reward
            try:
                qed = QED.qed(mol)
                metrics['qed_scores'].append(qed)
                # Sigmoid reward centered at min_qed
                qed_reward = 1.0 / (1.0 + np.exp(-10 * (qed - self.min_qed)))
                qed_rewards[i] = qed_reward
            except:
                metrics['qed_scores'].append(0.0)
                qed_rewards[i] = 0.0

            # BBB reward
            try:
                results = self.bbb_predictor.predict([smi])
                bbb_prob = float(results.ensemble_probability[0])
                metrics['bbb_probs'].append(bbb_prob)
                # Sigmoid reward centered at min_bbb_prob
                bbb_reward = 1.0 / (1.0 + np.exp(-10 * (bbb_prob - self.min_bbb_prob)))
                bbb_rewards[i] = bbb_reward
            except Exception as e:
                metrics['bbb_probs'].append(0.0)
                bbb_rewards[i] = 0.0

            # SA reward (lower is better)
            sa = compute_sa_score(smi)
            metrics['sa_scores'].append(sa)
            # Inverse reward: higher SA -> lower reward
            sa_reward = max(0.0, 1.0 - (sa / self.max_sa))
            sa_rewards[i] = sa_reward

        # Total reward
        total_reward = (
            self.bbb_weight * bbb_rewards +
            self.qed_weight * qed_rewards +
            self.sa_weight * sa_rewards +
            self.validity_weight * validity_rewards
        )

        return RewardOutput(
            total_reward=total_reward,
            bbb_reward=bbb_rewards,
            qed_reward=qed_rewards,
            sa_reward=sa_rewards,
            validity_reward=validity_rewards,
            metrics=metrics,
        )


def compute_rewards(
    smiles_list: List[str],
    bbb_predictor,
    cfg,
) -> RewardOutput:
    """
    Compute rewards for generated molecules.

    Args:
        smiles_list: List of SMILES strings
        bbb_predictor: BBB prediction model
        cfg: GAN configuration with reward weights

    Returns:
        RewardOutput with rewards and metrics
    """
    reward_net = RewardNetwork(
        bbb_predictor=bbb_predictor,
        bbb_weight=cfg.reward_bbb,
        qed_weight=cfg.reward_qed,
        sa_weight=cfg.reward_sa,
        validity_weight=cfg.reward_validity,
        min_qed=cfg.min_qed,
        min_bbb_prob=cfg.min_bbb_prob,
        max_sa=cfg.max_sa_score,
    )

    return reward_net(smiles_list)


def policy_gradient_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute policy gradient loss (REINFORCE).

    Args:
        log_probs: Log probabilities [batch_size, seq_len]
        rewards: Rewards [batch_size]
        baseline: Baseline for variance reduction [batch_size]

    Returns:
        Loss scalar
    """
    if baseline is not None:
        advantages = rewards - baseline
    else:
        advantages = rewards

    # Negative log likelihood weighted by advantage
    loss = -(log_probs * advantages.unsqueeze(-1)).sum(dim=1).mean()

    return loss


class BaselineNetwork(nn.Module):
    """
    Baseline network for variance reduction in policy gradient.

    Predicts expected reward from latent vectors.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict baseline reward.

        Args:
            z: Latent vectors [batch_size, latent_dim]

        Returns:
            Baseline rewards [batch_size, 1]
        """
        return self.net(z).squeeze(-1)
