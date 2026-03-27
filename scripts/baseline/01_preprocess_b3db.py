#!/usr/bin/env python
"""
B3DB Data Preprocessing

This script:
1. Loads B3DB classification/regression datasets
2. Filters groups and canonicalizes SMILES
3. Removes invalid molecules and duplicates
4. Performs scaffold-based splitting
5. Saves train/val/test splits

Usage:
    python scripts/baseline/01_preprocess_b3db.py --seed 0 --groups A,B --split_type scaffold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import B3DBPreprocessor, scaffold_split, random_split, B3DBDataset
from src.utils.io import write_csv
from src.config import Paths


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess B3DB dataset and create train/val/test splits"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for splitting"
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="A,B",
        help="Comma-separated list of groups to keep (e.g., 'A,B' or 'A,B,C')"
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="scaffold",
        choices=["scaffold", "random"],
        help="Type of split to perform"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Task type"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data/splits/seed_{seed}/{task}_{split_type})"
    )

    args = parser.parse_args()

    # Parse groups
    groups = tuple(g.strip() for g in args.groups.split(",") if g.strip())

    # Setup paths
    P = Paths()

    # Initialize preprocessor
    preprocessor = B3DBPreprocessor()

    # Load data
    print(f"Loading B3DB {args.task} dataset...")
    if args.task == "classification":
        filepath = P.data_raw / "B3DB_classification.tsv"
        data = preprocessor.load_classification(
            filepath=filepath,
            groups=groups,
            deduplicate=True,
            canonicalize=True,
        )
    else:
        filepath = P.data_raw / "B3DB_regression.tsv"
        data = preprocessor.load_regression(
            filepath=filepath,
            groups=groups,
            deduplicate=True,
            canonicalize=True,
        )

    print(f"Loaded {len(data)} samples")

    # Get statistics
    stats = preprocessor.get_statistics(data)
    print(f"Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Perform split
    print(f"\nPerforming {args.split_type} split...")
    if args.split_type == "scaffold":
        split = scaffold_split(
            df=data.df,
            label_col="y_cls" if args.task == "classification" else "logBB",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        split = random_split(
            df=data.df,
            label_col="y_cls" if args.task == "classification" else "logBB",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    print(f"Split sizes:")
    print(f"  Train: {len(split.train)}")
    print(f"  Val: {len(split.val)}")
    print(f"  Test: {len(split.test)}")

    # Create dataset
    dataset = B3DBDataset(
        train_df=split.train,
        val_df=split.val,
        test_df=split.test,
        task=args.task,
    )

    # Save splits
    if args.output_dir is None:
        output_dir = P.data_splits / f"seed_{args.seed}" / f"{args.task}_{args.split_type}"
    else:
        output_dir = Path(args.output_dir)

    print(f"\nSaving splits to {output_dir}...")
    dataset.save_splits(output_dir)

    # Save statistics
    stats_output = output_dir / "statistics.json"
    import json
    import numpy as np

    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        else:
            return obj

    stats_json = convert_numpy(stats)
    with open(stats_output, "w") as f:
        json.dump(stats_json, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
