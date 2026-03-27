"""
GAN (Generative Adversarial Network) module for molecule generation.

This module implements MolGAN with reinforcement learning for generating
BBB-permeable molecules. The GAN framework consists of:
- Generator: Creates molecular graphs from noise
- Discriminator: Distinguishes real from generated molecules
- Reward Network: Provides RL signals based on BBB prediction, QED, SA

Key components:
- MolGAN: Main GAN model
- GraphGenerator: Generator network
- GraphDiscriminator: Discriminator network
- RewardNetwork: Multi-objective reward computation
"""

from .molgan import MolGAN, GraphGenerator, GraphDiscriminator
from .reward import RewardNetwork, compute_rewards, RewardOutput
from .train_molgan import MolGANTrainer, train_gan, generate_with_gan

__all__ = [
    'MolGAN',
    'GraphGenerator',
    'GraphDiscriminator',
    'RewardNetwork',
    'compute_rewards',
    'RewardOutput',
    'MolGANTrainer',
    'train_gan',
    'generate_with_gan',
]
