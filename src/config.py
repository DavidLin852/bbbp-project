from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = root / "data" / "raw"
    data_splits: Path = root / "data" / "splits"
    data_external: Path = root / "data" / "external"
    artifacts: Path = root / "artifacts"
    features: Path = artifacts / "features"
    models: Path = artifacts / "models"
    metrics: Path = artifacts / "metrics"
    figures: Path = artifacts / "figures"
    logs: Path = artifacts / "logs"

@dataclass(frozen=True)
class DatasetConfig:
    filename: str = "B3DB_classification.tsv"
    group_keep: tuple[str, ...] = ("A", "B")
    smiles_col: str = "SMILES"
    group_col: str = "group"
    bbb_col: str = "BBB+/BBB-"
    logbb_col: str = "logBB"
    id_cols: tuple[str, ...] = ("NO.", "CID", "compound_name")

@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

@dataclass(frozen=True)
class FeaturizeConfig:
    # Morgan fingerprint
    morgan_bits: int = 2048
    morgan_radius: int = 2
    # MACCS keys
    maccs_bits: int = 167
    # Atom pairs
    atom_pairs_bits: int = 1024
    atom_pairs_max_dist: int = 3
    # FP2
    fp2_bits: int = 2048
    # Descriptor set: "basic" (13), "extended" (~100), "all" (200+)
    descriptor_set: str = "all"
    # Combine fingerprints + descriptors
    combine_features: bool = True

@dataclass(frozen=True)
class FingerprintConfig:
    """Configuration for fingerprint feature extraction."""
    morgan_bits: int = 2048
    morgan_radius: int = 2
    maccs_bits: int = 167
    atom_pairs_bits: int = 1024
    atom_pairs_max_dist: int = 3
    fp2_bits: int = 2048

@dataclass(frozen=True)
class DescriptorConfig:
    """Configuration for descriptor extraction."""
    set: str = "all"  # "basic", "extended", "all"
    normalize: bool = True

@dataclass(frozen=True)
class FeatureConfig:
    """Combined feature configuration."""
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)
    # Feature combinations
    use_morgan: bool = True
    use_maccs: bool = True
    use_atom_pairs: bool = True
    use_fp2: bool = True
    use_descriptors: bool = True

@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for Transformer model."""
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

@dataclass(frozen=True)
class StackingConfig:
    """Configuration for Stacking ensemble."""
    base_estimators: tuple[str, ...] = ("rf", "xgb", "lgbm")
    final_estimator: str = "lr"  # "lr" for LogisticRegression
    cv: int = 5
    passthrough: bool = True

@dataclass(frozen=True)
class SHAPConfig:
    """Configuration for SHAP analysis."""
    n_background_samples: int = 100
    n_test_samples: int = 100
    feature_names: list[str] = field(default_factory=list)
    plot_type: str = "summary"  # "summary", "dependence", "force"
    output_dir: Path = Path(__file__).resolve().parents[1] / "outputs" / "shap"


# ==================== VAE Configuration ====================

@dataclass(frozen=True)
class VAEConfig:
    """Configuration for Molecule VAE model."""
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
    """Configuration for VAE training process."""
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
    vae_model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "artifacts" / "models" / "vae")
    vae_logs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "artifacts" / "logs" / "vae")


# ==================== GAN Configuration ====================

@dataclass(frozen=True)
class GANConfig:
    """Configuration for MolGAN model."""
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
    """Configuration for GAN training process."""
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
    gan_model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "artifacts" / "models" / "gan")
    gan_logs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "artifacts" / "logs" / "gan")


# ==================== Generation Configuration ====================

@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for molecule generation pipeline."""
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
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "outputs" / "generated_molecules")

    # Model paths
    vae_model_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "artifacts" / "models" / "vae" / "best.pt")
    gan_model_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "artifacts" / "models" / "gan" / "best.pt")
    bbb_predictor_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "artifacts" / "models" / "seed_0_full" / "ensemble")
