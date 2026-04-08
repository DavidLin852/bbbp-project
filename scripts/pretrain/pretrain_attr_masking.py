#!/usr/bin/env python
"""
Pretrain GNN with Node Attribute Masking (Strategy 1)

Based on Hu et al. 2019: randomly mask 15% of node features,
reconstruct with node-level decoder.

Usage:
    python scripts/pretrain/pretrain_attr_masking.py \
        --num_samples 100000 --epochs 10 --batch_size 256 --model gin
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pretrain.attr_masking import pretrain_attr_masking


def parse_args():
    parser = argparse.ArgumentParser(description="GNN Pretraining: Node Attribute Masking")

    parser.add_argument("--data_dir", type=str, default="data/zinc22")
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="gin", choices=["gin", "gat", "gcn"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--save_dir", type=str, default="artifacts/models/pretrain/attr_masking")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=100)

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("GNN Pretraining: Node Attribute Masking (Strategy 1)")
    print("=" * 60)
    print(f"Model: {args.model.upper()} ({args.hidden_dim}d, {args.num_layers}layers)")
    print(f"Data: {args.data_dir}, Samples: {args.num_samples:,}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Mask ratio: {args.mask_ratio}")
    print("=" * 60)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: ZINC22 directory not found: {data_dir}")
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = pretrain_attr_masking(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        mask_ratio=args.mask_ratio,
        save_dir=args.save_dir,
        device=args.device,
        gradient_accumulation=args.gradient_accumulation,
        log_interval=args.log_interval,
    )

    print("\n" + "=" * 60)
    print("Pretraining Complete!")
    print("=" * 60)
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    print(f"Saved to: {args.save_dir}")
    print(f"Backbone: {save_dir}/{args.model}_pretrained_backbone.pt")
    print("\nFine-tune on B3DB:")
    print(f"  python scripts/gnn/run_gnn_benchmark.py \\")
    print(f"    --pretrained_encoder {save_dir}/{args.model}_pretrained_backbone.pt \\")
    print(f"    --tasks classification")
    print("=" * 60)


if __name__ == "__main__":
    main()
