#!/usr/bin/env python
"""Pretrain GNN with Edge Feature Prediction (Strategy 2)"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pretrain.edge_masking import pretrain_edge_masking

def parse_args():
    parser = argparse.ArgumentParser(description="GNN: Edge Feature Prediction")
    parser.add_argument("--data_dir", type=str, default="data/zinc22")
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="gin", choices=["gin", "gat", "gcn"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--edge_drop_ratio", type=float, default=0.15)
    parser.add_argument("--save_dir", type=str, default="artifacts/models/pretrain/edge_masking")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print("GNN Pretraining: Edge Feature Prediction (Strategy 2)")
    print("=" * 60)
    print(f"Model: {args.model.upper()}, edge_drop_ratio={args.edge_drop_ratio}")
    print(f"Samples: {args.num_samples:,}, Epochs: {args.epochs}")
    print("=" * 60)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = pretrain_edge_masking(
        data_dir=args.data_dir, num_samples=args.num_samples, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr, model_type=args.model, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, heads=args.heads, edge_drop_ratio=args.edge_drop_ratio,
        save_dir=args.save_dir, device=args.device,
        gradient_accumulation=args.gradient_accumulation, log_interval=args.log_interval,
    )
    print(f"\nBackbone: {save_dir}/{args.model}_pretrained_backbone.pt")

if __name__ == "__main__":
    main()
