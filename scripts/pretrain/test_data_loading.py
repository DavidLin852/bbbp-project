#!/usr/bin/env python
"""
Quick test script for ZINC22 data loading.

Verifies that the ZINC22 data can be loaded correctly.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pretrain.data import ZINC22Dataset, count_zinc22_molecules


def main():
    print("=" * 60)
    print("Testing ZINC22 Data Loading")
    print("=" * 60)

    # Test 1: Count total molecules
    print("\nTest 1: Counting molecules in data/zinc22/")
    try:
        total = count_zinc22_molecules("data/zinc22")
        print(f"✅ Total molecules: {total:,}")
    except Exception as e:
        print(f"❌ Count failed: {e}")
        return

    # Test 2: Load small dataset (graph)
    print("\nTest 2: Loading graph dataset (1K samples)")
    try:
        dataset = ZINC22Dataset(
            data_dir="data/zinc22",
            representation="graph",
            num_samples=1000,
        )
        print(f"✅ Loaded {len(dataset)} samples")

        # Test first sample
        sample = dataset[0]
        print(f"   Sample 0: {sample.num_nodes} nodes, {sample.edge_index.shape[1]} edges")
    except Exception as e:
        print(f"❌ Graph loading failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Load small dataset (SMILES)
    print("\nTest 3: Loading SMILES dataset (1K samples)")
    try:
        dataset = ZINC22Dataset(
            data_dir="data/zinc22",
            representation="smiles",
            num_samples=1000,
        )
        print(f"✅ Loaded {len(dataset)} samples")

        # Test first sample
        sample = dataset[0]
        print(f"   Sample 0: {sample[:50]}...")
    except Exception as e:
        print(f"❌ SMILES loading failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Load larger dataset (10K)
    print("\nTest 4: Loading larger dataset (10K samples)")
    try:
        dataset = ZINC22Dataset(
            data_dir="data/zinc22",
            representation="graph",
            num_samples=10000,
        )
        print(f"✅ Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Large dataset loading failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
