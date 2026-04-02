#!/usr/bin/env python
"""
Feature Computation

This script computes molecular features for preprocessed splits:
- Fingerprints (Morgan, MACCS, AtomPairs, FP2, Combined)
- Physicochemical descriptors

Usage:
    python scripts/baseline/02_compute_features.py --seed 0 --split scaffold --feature morgan
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import Paths
from src.data import B3DBDataset
from src.features import FingerprintGenerator, DescriptorGenerator
from src.utils.io import write_csv


def main():
    parser = argparse.ArgumentParser(
        description="Compute molecular features for B3DB splits"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="scaffold",
        choices=["scaffold", "random"],
        help="Split type"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Task type"
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="morgan",
        choices=[
            "morgan", "maccs", "atom_pairs", "fp2", "combined",
            "descriptors_basic", "descriptors_extended", "descriptors_all"
        ],
        help="Feature type to compute"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: artifacts/features/seed_{seed}/{split}/{feature})"
    )

    args = parser.parse_args()

    # Setup paths
    P = Paths()

    # Load splits
    split_dir = P.data_splits / f"seed_{args.seed}" / f"{args.task}_{args.split}"
    print(f"Loading splits from {split_dir}...")

    dataset = B3DBDataset.load_splits(
        split_dir,
        task=args.task,
    )

    print(f"Train: {len(dataset.train)}, Val: {len(dataset.val)}, Test: {len(dataset.test)}")

    # Compute features
    print(f"\nComputing {args.feature} features...")

    if args.feature.startswith("descriptors"):
        # Descriptor features
        descriptor_set = args.feature.split("_")[1]  # basic, extended, all
        generator = DescriptorGenerator(descriptor_set=descriptor_set)

        train_features = generator.compute(dataset.train[dataset.smiles_col].tolist())
        val_features = generator.compute(dataset.val[dataset.smiles_col].tolist())
        test_features = generator.compute(dataset.test[dataset.smiles_col].tolist())

        # Normalize
        train_features_norm = generator.fit_normalize(train_features)
        val_features_norm = generator.normalize(val_features)
        test_features_norm = generator.normalize(test_features)

        # Convert to numpy
        X_train = train_features_norm.to_numpy()
        X_val = val_features_norm.to_numpy()
        X_test = test_features_norm.to_numpy()

    else:
        # Fingerprint features
        generator = FingerprintGenerator()

        X_train = generator.compute(
            dataset.train[dataset.smiles_col].tolist(),
            fingerprint_type=args.feature,
        )
        X_val = generator.compute(
            dataset.val[dataset.smiles_col].tolist(),
            fingerprint_type=args.feature,
        )
        X_test = generator.compute(
            dataset.test[dataset.smiles_col].tolist(),
            fingerprint_type=args.feature,
        )

        # Convert sparse to dense if needed
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
            X_val = X_val.toarray()
            X_test = X_test.toarray()

    print(f"Feature shape: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # Save features (shared path; label files are task-specific)
    if args.output_dir is None:
        output_dir = P.features / f"seed_{args.seed}" / args.split / args.feature
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving features to {output_dir}...")

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "X_test.npy", X_test)

    # Save labels (task-specific filenames to avoid collision)
    if args.task == "classification":
        y_train = dataset.train[dataset.label_col].to_numpy()
        y_val = dataset.val[dataset.label_col].to_numpy()
        y_test = dataset.test[dataset.label_col].to_numpy()
    else:
        y_train = dataset.train["logBB"].to_numpy()
        y_val = dataset.val["logBB"].to_numpy()
        y_test = dataset.test["logBB"].to_numpy()

    np.save(output_dir / f"y_{args.task}_train.npy", y_train)
    np.save(output_dir / f"y_{args.task}_val.npy", y_val)
    np.save(output_dir / f"y_{args.task}_test.npy", y_test)

    # Save metadata
    import json
    metadata = {
        "seed": args.seed,
        "split_type": args.split,
        "task": args.task,
        "feature_type": args.feature,
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
