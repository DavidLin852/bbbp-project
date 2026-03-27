"""
Script 08: Train MolGAN for molecule generation.

Trains a MolGAN model with reinforcement learning for generating
BBB-permeable molecules.

Usage:
    python scripts/08_train_gan.py --seed 0 --epochs 200 --dataset A,B

    # With RL fine-tuning
    python scripts/08_train_gan.py --seed 0 --rl_start 50 --rl_weight 0.5
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import GANConfig, GANTrainConfig, DatasetConfig, Paths
from src.gan import train_gan
from src.vae.dataset import BBBDataset, create_train_val_splits
from src.featurize.graph_pyg import GraphBuildConfig
from src.multi_model_predictor import MultiModelPredictor, EnsembleStrategy
from src.utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train MolGAN for molecule generation")

    # General
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    # Dataset
    parser.add_argument("--dataset", type=str, default="A,B", help="B3DB groups (comma-separated)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to use")

    # Training
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--g_lr", type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument("--d_lr", type=float, default=1e-4, help="Discriminator learning rate")

    # RL fine-tuning
    parser.add_argument("--rl_start", type=int, default=50, help="Epoch to start RL fine-tuning")
    parser.add_argument("--rl_weight", type=float, default=0.5, help="RL reward weight")
    parser.add_argument("--n_critic", type=int, default=5, help="Discriminator updates per generator")

    # GAN config
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")

    # Output
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    seed_everything(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    paths = Paths()
    dataset_cfg = DatasetConfig()

    # Prepare datasets
    print("Loading B3DB dataset...")
    b3db_path = paths.data_raw / dataset_cfg.filename
    groups = [g.strip() for g in args.dataset.split(",")]

    # Load and filter data
    import pandas as pd
    df = pd.read_csv(b3db_path, sep="\t")
    df = df[df[dataset_cfg.group_col].isin(groups)].reset_index(drop=True)
    df["y_cls"] = df[dataset_cfg.bbb_col].map({"BBB+": 1, "BBB-": 0})
    df = df[df["y_cls"] == 1].reset_index(drop=True)

    if args.max_samples:
        df = df.head(args.max_samples)

    print(f"Loaded {len(df)} BBB+ molecules")

    # Create train/val splits
    train_smiles, val_smiles, test_smiles = create_train_val_splits(
        df["SMILES"].tolist(),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed,
    )

    print(f"Train: {len(train_smiles)}, Val: {len(val_smiles)}, Test: {len(test_smiles)}")

    # Create graph datasets (simplified - use MoleculeDataset for GAN)
    # In practice, you'd want proper graph data
    from src.vae.dataset import SMILESDataset

    train_ds = SMILESDataset(train_smiles)
    val_ds = SMILESDataset(val_smiles)

    print(f"GAN datasets: train={len(train_ds)}, val={len(val_ds)}")

    # Create GAN config
    gan_cfg = GANConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=3,
        gat_heads=4,
        dropout=0.3,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        epochs=args.epochs,
        n_critic=args.n_critic,
        rl_start_epoch=args.rl_start,
        rl_weight=args.rl_weight,
    )

    # Create training config
    train_cfg = GANTrainConfig(
        seed=args.seed,
        device=str(device),
        warmup_epochs=args.rl_start,
        rl_epochs=args.epochs - args.rl_start,
        gan_model_dir=Path(args.output_dir) if args.output_dir else paths.artifacts / "models" / "gan",
        gan_logs_dir=paths.artifacts / "logs" / "gan",
    )

    # Load BBB predictor for RL rewards
    print("Loading BBB predictor...")
    bbb_predictor = MultiModelPredictor(
        seed=args.seed,
        strategy=EnsembleStrategy.SOFT_VOTING,
    )

    # Train GAN
    print("Training MolGAN...")
    trainer = train_gan(
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=train_cfg,
        gan_cfg=gan_cfg,
        bbb_predictor=bbb_predictor,
    )

    print("Training complete!")
    print(f"Model saved to: {train_cfg.gan_model_dir}")
    print(f"Best score: {trainer.best_score:.4f}")


if __name__ == "__main__":
    main()
