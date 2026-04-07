#!/usr/bin/env python
"""
Graph Pretraining Script

Pretrain GNN models (GIN/GAT) on ZINC22 molecular graphs.
Uses molecular property prediction (logP, TPSA, MW, rotatable bonds).

Usage:
    # Quick test (100K molecules, 5 epochs)
    python scripts/pretrain/pretrain_graph.py \
        --num_samples 100000 --epochs 5 --batch_size 256

    # Full pretraining (2M molecules, 20 epochs)
    python scripts/pretrain/pretrain_graph.py \
        --num_samples 2000000 --epochs 20 --batch_size 256 \
        --save_dir artifacts/models/pretrain/gin_full

    # Multi-GPU (2 GPUs)
    python -m torch.distributed.run --nproc_per_node=2 \
        scripts/pretrain/pretrain_graph.py \
        --num_samples 2000000 --epochs 20 --batch_size 256
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pretrain.graph_trainer import pretrain_gnn_model


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain GNN on ZINC22")

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/zinc22",
        help="Path to ZINC22 directory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100000,
        help="Number of molecules to pretrain on",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers for parallel graph construction",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="gin",
        choices=["gin", "gat", "gcn"],
        help="GNN model type",
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
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Number of attention heads (GAT only)",
    )

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
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
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Logging interval (steps)",
    )

    # Output
    parser.add_argument(
        "--save_dir",
        type=str,
        default="artifacts/models/pretrain",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/cpu/cuda)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Graph Pretraining on ZINC22")
    print("=" * 60)
    print(f"Model: {args.model.upper()} ({args.hidden_dim}d, {args.num_layers}layers)")
    print(f"Data: {args.data_dir}")
    print(f"Samples: {args.num_samples:,}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("=" * 60)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\nError: ZINC22 directory not found: {data_dir}")
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting pretraining...\n")

    history = pretrain_gnn_model(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        device=args.device,
        gradient_accumulation=args.gradient_accumulation,
        log_interval=args.log_interval,
    )

    print("\n" + "=" * 60)
    print("Pretraining Complete!")
    print("=" * 60)
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("\nPretrained backbone:")
    print(f"  {save_dir}/{args.model}_pretrained_backbone.pt")
    print("\nTo fine-tune on B3DB:")
    print(f"  python scripts/gnn/run_gnn_benchmark.py \\")
    print(f"    --pretrained_encoder {save_dir}/{args.model}_pretrained_backbone.pt \\")
    print(f"    --tasks classification")
    print("=" * 60)


if __name__ == "__main__":
    main()
