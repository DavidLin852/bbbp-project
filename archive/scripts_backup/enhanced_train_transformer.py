"""
Transformer Model Training Script

使用Transformer架构训练分子指纹分类器。

Usage:
    # 基础训练
    python scripts/enhanced_train_transformer.py --seed 0 --features morgan

    # 自定义参数
    python scripts/enhanced_train_transformer.py --seed 0 --features combined \
        --hidden_dim 512 --num_layers 6 --num_heads 8 --epochs 200

    # 组合特征
    python scripts/enhanced_train_transformer.py --seed 0 --features combined \
        --batch_size 32 --lr 5e-4
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import sparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Paths, DatasetConfig, TransformerConfig
from src.transformer.transformer_model import (
    TransformerClassifier, train_transformer, evaluate_transformer
)


def load_features(seed: int, feature_type: str):
    """Load features for training."""
    P = Paths()
    feat_dir = P.features / f"seed_{seed}_enhanced"

    if feature_type == "combined":
        X = sparse.load_npz(feat_dir / "combined_all.npz")
    elif feature_type == "morgan":
        X = sparse.load_npz(feat_dir / "morgan.npz")
    elif feature_type == "maccs":
        X = sparse.load_npz(feat_dir / "maccs.npz")
    elif feature_type == "atompairs":
        X = sparse.load_npz(feat_dir / "atompairs.npz")
    elif feature_type == "fp2":
        X = sparse.load_npz(feat_dir / "fp2.npz")
    elif feature_type == "descriptors":
        X = sparse.load_npz(feat_dir / "descriptors.npz")
    else:
        fp_file = feat_dir / f"{feature_type}.npz"
        X = sparse.load_npz(fp_file)

    # Load meta mapping
    meta = pd.read_csv(feat_dir / "meta.csv")
    row_id_to_idx = {row_id: idx for idx, row_id in enumerate(meta["row_id"].values)}

    return X, row_id_to_idx


def load_splits(seed: int):
    """Load train/val/test splits."""
    P = Paths()
    split_dir = P.data_splits / f"seed_{seed}"

    df_train = pd.read_csv(split_dir / "train.csv")
    df_val = pd.read_csv(split_dir / "val.csv")
    df_test = pd.read_csv(split_dir / "test.csv")

    return df_train, df_val, df_test


def main():
    ap = argparse.ArgumentParser(description="Train Transformer model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", type=str, default="morgan",
                    choices=["morgan", "maccs", "atompairs", "fp2", "descriptors", "combined"],
                    help="Feature type to use")
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--feedforward_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience")
    ap.add_argument("--save_all", action="store_true",
                    help="Save all checkpoints")

    args = ap.parse_args()

    P = Paths()
    D = DatasetConfig()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Transformer Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Features: {args.features}")

    # Load data
    print("\nLoading data...")
    X_full, row_id_to_idx = load_features(args.seed, args.features)
    df_train, df_val, df_test = load_splits(args.seed)

    # Get indices using the mapping
    train_idx = df_train["row_id"].map(row_id_to_idx).values
    val_idx = df_val["row_id"].map(row_id_to_idx).values
    test_idx = df_test["row_id"].map(row_id_to_idx).values

    # Extract splits
    X_train = X_full[train_idx]
    X_val = X_full[val_idx]
    X_test = X_full[test_idx]

    y_train = df_train["y_cls"].values.astype(np.float32)
    y_val = df_val["y_cls"].values.astype(np.float32)
    y_test = df_test["y_cls"].values.astype(np.float32)

    # Convert to dense if sparse
    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray().astype(np.float32)
        X_val = X_val.toarray().astype(np.float32)
        X_test = X_test.toarray().astype(np.float32)

    print(f"Train: {X_train.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    input_dim = X_train.shape[1]
    model = TransformerClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout
    )

    print(f"\nModel Architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Heads: {args.num_heads}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training config
    config = TransformerConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.patience
    )

    # Output directory
    model_dir = P.models / f"seed_{args.seed}_enhanced" / args.features
    model_dir.mkdir(parents=True, exist_ok=True)

    # Train
    print(f"\n{'='*60}")
    print("Training...")
    print(f"{'='*60}")

    history = train_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=str(model_dir)
    )

    # Load best model
    model.load_state_dict(torch.load(f"{model_dir}/best_model.pt"))
    print(f"\nLoaded best model (epoch {history['best_epoch']}, AUC: {history['best_auc']:.4f})")

    # Evaluate
    print(f"\n{'='*60}")
    print("Evaluation on Test Set")
    print(f"{'='*60}")

    results = evaluate_transformer(model, test_loader, device)

    print(f"\nMetrics:")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  AUPRC: {results['auprc']:.4f}")

    # Save results
    results_json = {
        "seed": args.seed,
        "features": args.features,
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "feedforward_dim": args.feedforward_dim,
            "dropout": args.dropout
        },
        "metrics": {
            "auc": results['auc'],
            "accuracy": results['accuracy'],
            "precision": results['precision'],
            "recall": results['recall'],
            "f1": results['f1'],
            "auprc": results['auprc']
        },
        "training": {
            "best_epoch": history['best_epoch'],
            "best_val_auc": history['best_auc'],
            "train_loss_history": history['train_loss'],
            "val_auc_history": history['val_auc']
        }
    }

    results_path = model_dir / "transformer_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Save training history
    history_path = model_dir / "training_history.csv"
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)
    print(f"History saved: {history_path}")

    # Save model
    torch.save(model.state_dict(), model_dir / "final_model.pt")
    print(f"Model saved: {model_dir / 'final_model.pt'}")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
