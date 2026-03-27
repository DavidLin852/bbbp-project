"""
Script 07: Train VAE for molecule generation.

Trains a MoleculeVAE model on BBB+ molecules from B3DB database.
Can optionally pre-train on ZINC molecules first.

Usage:
    python scripts/07_train_vae.py --seed 0 --epochs 200 --dataset A,B

    # Pre-train on ZINC, then fine-tune on B3DB
    python scripts/07_train_vae.py --seed 0 --pretrain --pretrain_epochs 100
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import VAEConfig, VAETrainConfig, DatasetConfig, Paths
from src.vae import train_vae, BBBDataset, MoleculeDataset
from src.vae.dataset import create_train_val_splits
from src.featurize.graph_pyg import GraphBuildConfig
from src.multi_model_predictor import MultiModelPredictor, EnsembleStrategy
from src.utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE for molecule generation")

    # General
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    # Dataset
    parser.add_argument("--dataset", type=str, default="A,B", help="B3DB groups (comma-separated)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to use")

    # Pre-training
    parser.add_argument("--pretrain", action="store_true", help="Pre-train on ZINC first")
    parser.add_argument("--pretrain_epochs", type=int, default=100, help="Pre-training epochs")
    parser.add_argument("--zinc_samples", type=int, default=100000, help="ZINC samples for pre-training")

    # Training
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # VAE config
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--beta", type=float, default=0.001, help="KL divergence weight")

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
    df = df[df["y_cls"] == 1].reset_index(drop=True)  # Only BBB+ for VAE

    if args.max_samples:
        df = df.head(args.max_samples)

    print(f"Loaded {len(df)} BBB+ molecules")

    # Create train/val splits
    train_df, val_df, test_df = create_train_val_splits(
        df["SMILES"].tolist(),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed,
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Create graph datasets
    cfg = GraphBuildConfig(
        smiles_col="SMILES",
        label_col="y_cls",
        id_col="row_id"
    )

    # Add row_id and y_cls to dataframes
    train_df = pd.DataFrame({"SMILES": train_df, "y_cls": [1] * len(train_df), "row_id": range(len(train_df))})
    val_df = pd.DataFrame({"SMILES": val_df, "y_cls": [1] * len(val_df), "row_id": range(len(val_df))})
    test_df = pd.DataFrame({"SMILES": test_df, "y_cls": [1] * len(test_df), "row_id": range(len(test_df))})

    # Create datasets
    root = str(paths.artifacts / "features" / f"seed_{args.seed}_vae")
    train_ds = BBBDataset(root=root + "/train", df=train_df, cfg=cfg, bbb_only=True)
    val_ds = BBBDataset(root=root + "/val", df=val_df, cfg=cfg, bbb_only=True)
    test_ds = BBBDataset(root=root + "/test", df=test_df, cfg=cfg, bbb_only=True)

    print(f"Graph datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Create VAE config
    vae_cfg = VAEConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=3,
        gat_heads=4,
        dropout=0.2,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        beta=args.beta,
    )

    # Create training config
    train_cfg = VAETrainConfig(
        seed=args.seed,
        device=str(device),
        vae_model_dir=Path(args.output_dir) if args.output_dir else paths.artifacts / "models" / "vae",
        vae_logs_dir=paths.artifacts / "logs" / "vae",
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.epochs,
    )

    # Pre-training on ZINC (if requested)
    if args.pretrain:
        print("Pre-training on ZINC molecules (not implemented - skipping)")
        # In practice, you would:
        # 1. Load ZINC molecules
        # 2. Pre-train VAE on ZINC
        # 3. Fine-tune on B3DB

    # Load BBB predictor for auxiliary loss
    print("Loading BBB predictor...")
    bbb_predictor = MultiModelPredictor(
        seed=args.seed,
        strategy=EnsembleStrategy.SOFT_VOTING,
    )

    # Train VAE
    print("Training VAE...")
    trainer = train_vae(
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=train_cfg,
        vae_cfg=vae_cfg,
        bbb_predictor=bbb_predictor,
    )

    print("Training complete!")
    print(f"Model saved to: {train_cfg.vae_model_dir}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
