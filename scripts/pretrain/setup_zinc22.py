#!/usr/bin/env python
"""
ZINC22 Pretraining Setup Script

Helper script to set up ZINC22 data for pretraining.
Creates small samples and verifies data integrity.

Usage:
    # Setup from existing ZINC22 file
    python scripts/pretrain/setup_zinc22.py \\
        --source data/zinc22/smiles_full.txt \\
        --num_samples 10000

    # Verify setup
    python scripts/pretrain/setup_zinc22.py --verify
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pretrain.data import create_small_zinc22_sample, ZINC22PretrainDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Setup ZINC22 data for pretraining")

    parser.add_argument(
        "--source",
        type=str,
        default="data/zinc22",
        help="ZINC22 directory (contains H04/, H05/, etc.)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples for smoke test",
    )
    parser.add_argument(
        "--create_sample",
        action="store_true",
        help="Create small sample from source file",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing data files",
    )
    parser.add_argument(
        "--test_load",
        action="store_true",
        help="Test data loading pipeline",
    )

    return parser.parse_args()


def verify_setup():
    """Verify that ZINC22 data is properly set up."""
    print("=" * 60)
    print("Verifying ZINC22 Setup")
    print("=" * 60)

    # Check for ZINC22 directory
    zinc22_dir = Path("data/zinc22")
    if zinc22_dir.exists():
        print(f"✅ ZINC22 directory found: {zinc22_dir}")

        # Count .smi.gz files
        smi_files = list(zinc22_dir.rglob("*.smi.gz"))
        print(f"   Found {len(smi_files)} .smi.gz files")

        # List subdirectories
        subdirs = [d for d in zinc22_dir.iterdir() if d.is_dir()]
        print(f"   Subdirectories: {', '.join([d.name for d in subdirs[:5]])}{'...' if len(subdirs) > 5 else ''}")
    else:
        print(f"❌ ZINC22 directory not found: {zinc22_dir}")
        print(f"   Expected structure: data/zinc22/H04/*.smi.gz")

    # Check for cache directory
    cache_dir = Path("data/zinc22/cache")
    if cache_dir.exists():
        print(f"✅ Cache directory exists: {cache_dir}")
    else:
        print(f"ℹ️  Cache directory not found (will be created): {cache_dir}")

    print("=" * 60)


def test_data_loading():
    """Test data loading pipeline."""
    print("\n" + "=" * 60)
    print("Testing Data Loading")
    print("=" * 60)

    # Test graph loading
    print("\nTesting graph representation...")
    try:
        from src.pretrain.data import ZINC22Dataset
        dataset = ZINC22Dataset(
            data_dir="data/zinc22",
            representation="graph",
            num_samples=1000,
        )
        print(f"✅ Graph dataset loaded: {len(dataset)} samples")

        # Test first sample
        sample = dataset[0]
        print(f"   Sample shape: nodes={sample.x.shape[0]}, edges={sample.edge_index.shape[1]}")

    except Exception as e:
        print(f"❌ Graph loading failed: {e}")

    # Test SMILES loading
    print("\nTesting SMILES representation...")
    try:
        from src.pretrain.data import ZINC22Dataset
        dataset = ZINC22Dataset(
            data_dir="data/zinc22",
            representation="smiles",
            num_samples=1000,
        )
        print(f"✅ SMILES dataset loaded: {len(dataset)} samples")

        # Test first sample
        sample = dataset[0]
        print(f"   Sample SMILES: {sample[:50]}...")

    except Exception as e:
        print(f"❌ SMILES loading failed: {e}")

    print("=" * 60)


def main():
    args = parse_args()

    if args.verify:
        verify_setup()
        if args.test_load:
            test_data_loading()
        return

    if args.create_sample:
        print("=" * 60)
        print("Creating ZINC22 Sample")
        print("=" * 60)
        print(f"Source: {args.source}")
        print(f"Samples: {args.num_samples:,}")

        # Create sample
        output_path = create_small_zinc22_sample(
            output_path="data/zinc22/smiles_small.txt",
            num_samples=args.num_samples,
            source_dir=args.source,
        )

        print(f"\n✅ Created: {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Verify: python scripts/pretrain/setup_zinc22.py --verify")
        print(f"  2. Test pretraining:")
        print(f"     python scripts/pretrain/pretrain_graph.py \\")
        print(f"         --data_dir {args.source} \\")
        print(f"         --num_samples {args.num_samples} \\")
        print(f"         --epochs 5")
        print("=" * 60)
        return

    # Default: show help
    import inspect
    help_text = inspect.cleandoc("""
        ZINC22 Pretraining Setup

        Common workflows:

        1. Initial setup from existing ZINC22 file:
           python scripts/pretrain/setup_zinc22.py \\
               --source /path/to/zinc22/smiles.txt \\
               --create_sample \\
               --num_samples 10000

        2. Verify setup:
           python scripts/pretrain/setup_zinc22.py --verify

        3. Test data loading:
           python scripts/pretrain/setup_zinc22.py --verify --test_load
    """)
    print(help_text)


if __name__ == "__main__":
    main()
