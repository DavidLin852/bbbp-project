#!/usr/bin/env python
"""
Graph Pretraining Script

Pretrain GNN models (GIN/GAT) on ZINC22 molecular graphs.

Usage:
    # Smoke test (10K molecules, 5 epochs)
    python scripts/pretrain/pretrain_graph.py \\
        --data_path data/zinc22/smiles_small.txt \\
        --num_samples 10000 \\
        --epochs 5 \\
        --model_type gin

    # Full pretraining (1M molecules, 100 epochs)
    python scripts/pretrain/pretrain_graph.py \\
        --data_path data/zinc22/smiles.txt \\
        --num_samples 1000000 \\
        --epochs 100 \\
        --model_type gin
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pretrain.graph import pretrain_graph_model
from src.pretrain.data import create_small_zinc22_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain graph models on ZINC22")

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/zinc22",
        help="Path to ZINC22 directory (contains H04/, H05/, etc.)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to use (default: 10K for smoke test)",
    )
    parser.add_argument(
        "--create_sample",
        action="store_true",
        help="Create small sample from source file first",
    )

    # Model
    parser.add_argument(
        "--model_type",
        type=str,
        default="gin",
        choices=["gin", "gat"],
        help="Model type to pretrain",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of GNN layers",
    )

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    # Output
    parser.add_argument(
        "--save_dir",
        type=str,
        default="artifacts/models/pretrain/graph",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto/cpu/cuda)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Graph Pretraining on ZINC22")
    print("=" * 60)
    print(f"Model: {args.model_type.upper()}")
    print(f"Data: {args.data_dir}")
    print(f"Samples: {args.num_samples:,}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Check data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\nError: ZINC22 directory not found: {data_dir}")
        print(f"\nExpected structure:")
        print(f"  {args.data_dir}/")
        print(f"    ├── H04/")
        print(f"    │   ├── *.smi.gz")
        print(f"    ├── H05/")
        print(f"    └── ...")
        return

    # Run pretraining
    print("\nStarting pretraining...\n")

    history = pretrain_graph_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        pretraining_task="property_prediction",
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        save_dir=args.save_dir,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("Pretraining Complete!")
    print("=" * 60)
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("\nPretrained backbone:")
    print(f"  {args.save_dir}/{args.model_type}_pretrained_backbone.pt")
    print("\nTo fine-tune on B3DB:")
    print(f"  python scripts/pretrain/finetune_graph.py \\")
    print(f"    --pretrained_path {args.save_dir}/{args.model_type}_pretrained_backbone.pt \\")
    print(f"    --task classification")
    print("=" * 60)


if __name__ == "__main__":
    main()
