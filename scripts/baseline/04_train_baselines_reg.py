#!/usr/bin/env python
"""
Train Baseline Regression Models

This script trains baseline regression models on precomputed features.

Usage:
    python scripts/baseline/04_train_baselines_reg.py --seed 0 --feature morgan --models rf_reg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import Paths
from src.models import ModelFactory
from src.train import train_multiple_regression_models
from src.evaluate import RegModelComparison, compare_regression_models, generate_reg_report


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline regression models"
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
        "--feature",
        type=str,
        default="morgan",
        help="Feature type"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="rf_reg",
        help="Comma-separated list of models (e.g., 'rf_reg')"
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Feature directory (default: artifacts/features/seed_{seed}/{split}/{feature})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: artifacts/models/regression/seed_{seed}/{split}/{feature})"
    )

    args = parser.parse_args()

    # Parse models
    model_types = [m.strip() for m in args.models.split(",") if m.strip()]

    # Setup paths
    P = Paths()

    # Load features
    if args.feature_dir is None:
        feature_dir = P.features / f"seed_{args.seed}" / args.split / args.feature
    else:
        feature_dir = Path(args.feature_dir)

    print(f"Loading features from {feature_dir}...")

    X_train = np.load(feature_dir / "X_train.npy")
    X_val = np.load(feature_dir / "X_val.npy")
    X_test = np.load(feature_dir / "X_test.npy")
    y_train = np.load(feature_dir / "y_regression_train.npy")
    y_val = np.load(feature_dir / "y_regression_val.npy")
    y_test = np.load(feature_dir / "y_regression_test.npy")

    print(f"Data shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    # Create models
    print(f"\nCreating models: {model_types}")
    factory = ModelFactory()
    models = [factory.create_model(mt) for mt in model_types]

    # Setup output directory
    if args.output_dir is None:
        output_dir = P.models / "regression" / f"seed_{args.seed}" / args.split / args.feature
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Train models
    print(f"\nTraining {len(models)} models...")
    results = train_multiple_regression_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_type=args.feature,
        output_dir=output_dir,
    )

    # Create comparison
    comparison = compare_regression_models(results=results)

    # Print results
    print("\nResults:")
    df = comparison.to_dataframe()
    print(df[["model_name", "val_r2", "test_r2", "test_rmse", "test_mae"]])

    # Save comparison
    comparison.save(output_dir / "comparison.json")

    # Generate report
    report_dir = output_dir / "reports"
    generate_reg_report(comparison, output_dir=report_dir)

    print(f"\nResults saved to {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
