#!/usr/bin/env python
"""
Train Baseline Models (Classification and Regression)

This script trains all classical baseline models on precomputed features.
It supports both classification (rf, xgb, lgbm, svm, lr, knn) and
regression (rf_reg, xgb_reg, lgbm_reg, svm_reg, ridge, knn_reg).

Usage:
    # Classification
    python scripts/baseline/04_train_all.py --seed 0 --task classification --feature morgan --models rf,xgb,lgbm,svm,lr,knn

    # Regression
    python scripts/baseline/04_train_all.py --seed 0 --task regression --feature morgan --models rf_reg,xgb_reg,lgbm_reg,svm_reg,ridge,knn_reg
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
from src.train import train_multiple_models, train_multiple_regression_models
from src.evaluate import ModelComparison, compare_regression_models, generate_report, generate_reg_report


# Classification models (produce probability-based metrics)
CLS_MODELS = {"rf", "xgb", "lgbm", "svm", "lr", "knn", "nb", "gb", "ada", "etc"}

# Regression models (produce continuous predictions)
REG_MODELS = {"rf_reg", "xgb_reg", "lgbm_reg", "svm_reg", "ridge", "knn_reg"}


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline models (classification and regression)"
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
        help="Feature type"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="rf,xgb,lgbm",
        help="Comma-separated list of models"
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Feature directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory"
    )

    args = parser.parse_args()

    # Parse models
    model_types = [m.strip() for m in args.models.split(",") if m.strip()]

    # Determine task
    is_regression = args.task == "regression"

    # Validate model types against task
    if is_regression:
        for mt in model_types:
            if mt not in REG_MODELS:
                print(f"WARNING: '{mt}' is not a regression model, skipping.")
        model_types = [mt for mt in model_types if mt in REG_MODELS]
    else:
        for mt in model_types:
            if mt not in CLS_MODELS:
                print(f"WARNING: '{mt}' is not a classification model, skipping.")
        model_types = [mt for mt in model_types if mt in CLS_MODELS]

    if not model_types:
        print("ERROR: No valid models for the selected task.")
        sys.exit(1)

    # Setup paths
    P = Paths()

    # Load features (shared path with task-specific labels)
    if args.feature_dir is None:
        feature_dir = P.features / f"seed_{args.seed}" / args.split / args.feature
    else:
        feature_dir = Path(args.feature_dir)

    print(f"Loading features from {feature_dir}...")

    X_train = np.load(feature_dir / "X_train.npy")
    X_val = np.load(feature_dir / "X_val.npy")
    X_test = np.load(feature_dir / "X_test.npy")
    y_train = np.load(feature_dir / f"y_{args.task}_train.npy")
    y_val = np.load(feature_dir / f"y_{args.task}_val.npy")
    y_test = np.load(feature_dir / f"y_{args.task}_test.npy")

    print(f"Data shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    # Create models
    print(f"\nCreating {len(model_types)} {args.task} models: {model_types}")
    factory = ModelFactory()
    models = [factory.create_model(mt) for mt in model_types]

    # Setup output directory
    task_dir = "regression" if is_regression else "baselines"
    if args.output_dir is None:
        output_dir = P.models / task_dir / f"seed_{args.seed}" / args.split / args.feature
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    print(f"\nTraining...")
    if is_regression:
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
        comparison = compare_regression_models(results=results)

        print("\nResults:")
        df = comparison.to_dataframe()
        print(df[["model_name", "val_r2", "test_r2", "test_rmse", "test_mae"]])

        comparison.save(output_dir / "comparison.json")
        report_dir = output_dir / "reports"
        generate_reg_report(comparison, output_dir=report_dir)
    else:
        results = train_multiple_models(
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
        comparison = ModelComparison(results=results)

        print("\nResults:")
        df = comparison.to_dataframe()
        print(df[["model_name", "val_auc", "test_auc", "test_f1"]])

        comparison.save(output_dir / "comparison.json")
        report_dir = output_dir / "reports"
        generate_report(comparison, output_dir=report_dir)

    print(f"\nResults saved to {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
