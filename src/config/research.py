"""
Research module configuration for future experimental features.

This module contains configuration classes for experimental and research
features that are NOT part of the current working baseline pipeline.

These configurations are preserved for future research but are NOT actively
used in the baseline pipeline. They include:
- Transformer models (MolBERT, Graphormer)
- VAE/GAN molecule generation
- Ensemble methods (stacking, voting)
- SHAP interpretability analysis
- Active learning
- Transport mechanism prediction

WARNING: These configurations are for research purposes only.
The baseline pipeline does NOT use these configs.

Usage:
    from src.config.research import TransformerConfig, VAEConfig, GANConfig

    # Transformer model
    transformer_config = TransformerConfig(hidden_dim=256, num_layers=4)

    # VAE generation
    vae_config = VAEConfig(latent_dim=256, beta=0.001)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ==================== Transformer Configuration ====================

@dataclass(frozen=True)
class TransformerConfig:
    """
    Configuration for Transformer-based models (future work).

    This config is for models like MolBERT or Graphormer.
    Not currently used in the baseline pipeline.

    Attributes:
        hidden_dim: Hidden dimension for transformer layers
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        feedforward_dim: Dimension of feedforward network
        dropout: Dropout rate
        learning_rate: Learning rate for training
        weight_decay: Weight decay for regularization
        batch_size: Batch size for training
        epochs: Number of training epochs
        early_stopping_patience: Patience for early stopping
    """

    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    feedforward_dim: int = 512
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10


# ==================== VAE Configuration ====================

@dataclass(frozen=True)
class VAEConfig:
    """
    Configuration for molecule VAE (Variational Autoencoder) generation.

    This config is for VAE-based molecule generation models.
    Not currently used in the baseline pipeline.

    Attributes:
        latent_dim: Dimension of latent space
        hidden_dim: Hidden dimension for encoder/decoder
        num_layers: Number of GNN layers in encoder/decoder
        gat_heads: Number of attention heads in GAT layers
        dropout: Dropout rate
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        epochs: Number of training epochs
        grad_clip: Gradient clipping threshold
        beta: KL divergence weight (for beta-VAE)
        lambda_bbb: BBB prediction loss weight
        lambda_qed: QED score loss weight
        kl_anneal: Whether to use KL annealing
        anneal_epochs: Number of epochs for KL annealing
        min_qed: Minimum QED score for generated molecules
        min_bbb_prob: Minimum BBB probability for generated molecules
        max_sa_score: Maximum synthetic accessibility score (lower is better)
    """

    # Model architecture
    latent_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 3
    gat_heads: int = 4
    dropout: float = 0.2

    # Training parameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    grad_clip: float = 5.0

    # Loss weights
    beta: float = 0.001  # KL divergence weight (for beta-VAE)
    lambda_bbb: float = 0.1  # BBB prediction loss weight
    lambda_qed: float = 0.05  # QED score loss weight

    # KL annealing
    kl_anneal: bool = True
    anneal_epochs: int = 50

    # Generation parameters
    min_qed: float = 0.5
    min_bbb_prob: float = 0.7
    max_sa_score: float = 4.0  # Synthetic accessibility (lower is better)


@dataclass(frozen=True)
class VAETrainConfig:
    """
    Configuration for VAE training process (future work).

    This config is for VAE training workflow.
    Not currently used in the baseline pipeline.

    Attributes:
        seed: Random seed
        device: Device for training ("cuda" or "cpu")
        pretrain_epochs: Epochs for pre-training on ZINC
        pretrain_samples: Number of samples for pre-training
        finetune_epochs: Epochs for fine-tuning on B3DB BBB+ molecules
        val_every: Validate every N epochs
        save_every: Save checkpoint every N epochs
        vae_model_dir: Directory for VAE model checkpoints
        vae_logs_dir: Directory for VAE training logs
    """

    seed: int = 0
    device: str = "cuda"

    # Pre-training on ZINC
    pretrain_epochs: int = 100
    pretrain_samples: int = 100000

    # Fine-tuning on B3DB BBB+ molecules
    finetune_epochs: int = 100

    # Validation
    val_every: int = 5
    save_every: int = 10

    # Paths
    vae_model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "models" / "vae")
    vae_logs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "logs" / "vae")


# ==================== GAN Configuration ====================

@dataclass(frozen=True)
class GANConfig:
    """
    Configuration for MolGAN (molecule GAN) generation.

    This config is for GAN-based molecule generation models.
    Not currently used in the baseline pipeline.

    Attributes:
        latent_dim: Dimension of latent noise vector
        hidden_dim: Hidden dimension for generator/discriminator
        num_layers: Number of GNN layers
        gat_heads: Number of attention heads in GAT layers
        dropout: Dropout rate
        batch_size: Batch size for training
        g_lr: Learning rate for generator
        d_lr: Learning rate for discriminator
        epochs: Number of training epochs
        grad_clip: Gradient clipping threshold
        n_critic: Train critic n times per generator step (WGAN-GP)
        gp_weight: Gradient penalty weight (for WGAN-GP)
        rl_start_epoch: Start RL fine-tuning after this epoch
        rl_weight: Weight for RL reward
        temperature: Softmax temperature for sampling
        reward_bbb: Weight for BBB permeability reward
        reward_qed: Weight for QED score reward
        reward_sa: Weight for synthetic accessibility reward
        reward_validity: Weight for molecular validity reward
        min_qed: Minimum QED score for generated molecules
        min_bbb_prob: Minimum BBB probability for generated molecules
        max_sa_score: Maximum synthetic accessibility score
    """

    # Model architecture
    latent_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 3
    gat_heads: int = 4
    dropout: float = 0.3

    # Training parameters
    batch_size: int = 64
    g_lr: float = 1e-4
    d_lr: float = 1e-4
    epochs: int = 200
    grad_clip: float = 5.0

    # GAN training
    n_critic: int = 5  # Train critic n times per generator step
    gp_weight: float = 10.0  # Gradient penalty weight (for WGAN-GP)

    # RL fine-tuning
    rl_start_epoch: int = 50  # Start RL fine-tuning after this epoch
    rl_weight: float = 0.5  # Weight for RL reward
    temperature: float = 1.0  # Softmax temperature for sampling

    # Reward function weights
    reward_bbb: float = 1.0
    reward_qed: float = 0.3
    reward_sa: float = 0.3
    reward_validity: float = 1.0

    # Generation parameters
    min_qed: float = 0.5
    min_bbb_prob: float = 0.7
    max_sa_score: float = 4.0


@dataclass(frozen=True)
class GANTrainConfig:
    """
    Configuration for GAN training process (future work).

    This config is for GAN training workflow.
    Not currently used in the baseline pipeline.

    Attributes:
        seed: Random seed
        device: Device for training ("cuda" or "cpu")
        warmup_epochs: Epochs for warmup phase (pre-RL)
        rl_epochs: Epochs for RL fine-tuning phase
        val_every: Validate every N epochs
        save_every: Save checkpoint every N epochs
        gan_model_dir: Directory for GAN model checkpoints
        gan_logs_dir: Directory for GAN training logs
    """

    seed: int = 0
    device: str = "cuda"

    # Warmup phase (pre-RL)
    warmup_epochs: int = 50

    # RL fine-tuning phase
    rl_epochs: int = 150

    # Validation
    val_every: int = 5
    save_every: int = 10

    # Paths
    gan_model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "models" / "gan")
    gan_logs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "logs" / "gan")


# ==================== Generation Configuration ====================

@dataclass(frozen=True)
class GenerationConfig:
    """
    Configuration for molecule generation pipeline (future work).

    This config is for VAE/GAN-based molecule generation.
    Not currently used in the baseline pipeline.

    Attributes:
        strategy: Generation strategy ("vae", "gan", or "both")
        n_generate: Number of molecules to generate
        min_qed: Minimum QED score for filtering
        min_bbb_prob: Minimum BBB probability for filtering
        max_sa_score: Maximum synthetic accessibility score
        check_novelty: Exclude molecules in training set
        remove_duplicates: Remove duplicate molecules
        output_dir: Directory for generated molecules
        vae_model_path: Path to trained VAE model
        gan_model_path: Path to trained GAN model
        bbb_predictor_path: Path to trained BBB predictor
    """

    # Generation strategy: "vae", "gan", or "both"
    strategy: str = "both"

    # Number of molecules to generate
    n_generate: int = 1000

    # Filtering thresholds
    min_qed: float = 0.5
    min_bbb_prob: float = 0.7
    max_sa_score: float = 4.0

    # Novelty check (exclude molecules in training set)
    check_novelty: bool = True

    # Diversity filtering (remove duplicates)
    remove_duplicates: bool = True

    # Output paths
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "outputs" / "generated_molecules")

    # Model paths
    vae_model_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "models" / "vae" / "best.pt")
    gan_model_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "models" / "gan" / "best.pt")
    bbb_predictor_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "models" / "seed_0_full" / "ensemble")


# ==================== Ensemble Configuration ====================

@dataclass(frozen=True)
class StackingConfig:
    """
    Configuration for Stacking ensemble (future work).

    This config is for ensemble methods.
    Not currently used in the baseline pipeline.

    Attributes:
        base_estimators: Tuple of model names for base estimators
        final_estimator: Model name for final estimator ("lr" for LogisticRegression)
        cv: Number of cross-validation folds
        passthrough: Whether to pass original features to final estimator
    """

    base_estimators: tuple[str, ...] = ("rf", "xgb", "lgbm")
    final_estimator: str = "lr"  # "lr" for LogisticRegression
    cv: int = 5
    passthrough: bool = True


# ==================== Interpretability Configuration ====================

@dataclass(frozen=True)
class SHAPConfig:
    """
    Configuration for SHAP analysis (future work).

    This config is for SHAP interpretability analysis.
    Not currently used in the baseline pipeline.

    Attributes:
        n_background_samples: Number of background samples for SHAP
        n_test_samples: Number of test samples to explain
        feature_names: List of feature names (optional)
        plot_type: Type of SHAP plot ("summary", "dependence", "force")
        output_dir: Directory for SHAP outputs
    """

    n_background_samples: int = 100
    n_test_samples: int = 100
    feature_names: list[str] = field(default_factory=list)
    plot_type: str = "summary"  # "summary", "dependence", "force"
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "outputs" / "shap")
