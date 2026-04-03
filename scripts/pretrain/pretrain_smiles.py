#!/usr/bin/env python
"""
SMILES Transformer Pretraining Script

Pretrain Transformer on ZINC22 SMILES using masked language modeling.

Usage:
    # Smoke test (10K molecules, 5 epochs)
    python scripts/pretrain/pretrain_smiles.py \\
        --data_path data/zinc22/smiles_small.txt \\
        --num_samples 10000 \\
        --epochs 5

    # Full pretraining (1M molecules, 100 epochs)
    python scripts/pretrain/pretrain_smiles.py \\
        --data_path data/zinc22/smiles.txt \\
        --num_samples 1000000 \\
        --epochs 100
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pretrain.smiles import pretrain_smiles_model
from src.pretrain.data import create_small_zinc22_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Transformer on ZINC22")

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
        "--d_model",
        type=int,
        default=256,
        help="Model dimension",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask",
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
        default=1e-4,
        help="Learning rate",
    )

    # Output
    parser.add_argument(
        "--save_dir",
        type=str,
        default="artifacts/models/pretrain/transformer",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to save/load tokenizer",
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
    print("SMILES Transformer Pretraining on ZINC22")
    print("=" * 60)
    print(f"Model: Transformer ({args.d_model}d, {args.n_heads}heads, {args.n_layers}layers)")
    print(f"Data: {args.data_dir}")
    print(f"Samples: {args.num_samples:,}")
    print(f"Epochs: {args.epochs}")
    print(f"Mask ratio: {args.mask_ratio}")
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

    history = pretrain_smiles_model(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mask_ratio=args.mask_ratio,
        save_dir=args.save_dir,
        device=args.device,
        tokenizer_path=args.tokenizer_path,
    )

    print("\n" + "=" * 60)
    print("Pretraining Complete!")
    print("=" * 60)
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("\nPretrained encoder:")
    print(f"  {args.save_dir}/transformer_pretrained_encoder.pt")
    print("\nTo fine-tune on B3DB:")
    print(f"  python scripts/transformer/run_transformer_benchmark.py \\")
    print(f"    --pretrained_encoder {args.save_dir}/transformer_pretrained_encoder.pt \\")
    print(f"    --tasks classification")
    print("=" * 60)


if __name__ == "__main__":
    main()
