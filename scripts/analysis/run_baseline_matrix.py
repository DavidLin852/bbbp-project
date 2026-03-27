#!/usr/bin/env python
"""
Run complete baseline experiment matrix.

This script runs the full baseline experimental matrix:
- Multiple seeds
- Multiple split types
- Multiple features
- Multiple models

Usage:
    python scripts/analysis/run_baseline_matrix.py --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def run_command(cmd: list[str], description: str):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        return False

    print(f"✓ {description} completed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run complete baseline experiment matrix"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated list of seeds (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="scaffold",
        help="Comma-separated list of split types (e.g., 'scaffold,random')"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="morgan,combined,descriptors_basic",
        help="Comma-separated list of features"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="rf,xgb,lgbm",
        help="Comma-separated list of models"
    )
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Skip preprocessing if already done"
    )
    parser.add_argument(
        "--skip_features",
        action="store_true",
        help="Skip feature computation if already done"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing"
    )

    args = parser.parse_args()

    # Parse arguments
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    splits = [s.strip() for s in args.splits.split(',')]
    features = [f.strip() for f in args.features.split(',')]
    models = [m.strip() for m in args.models.split(',')]

    # Calculate total experiments
    n_experiments = len(seeds) * len(splits) * len(features) * len(models)

    print("="*60)
    print("BBB Baseline Experiment Matrix")
    print("="*60)
    print(f"Seeds: {seeds}")
    print(f"Splits: {splits}")
    print(f"Features: {features}")
    print(f"Models: {models}")
    print(f"Total experiments: {n_experiments}")
    print("="*60)

    if args.dry_run:
        print("\nDRY RUN - Commands will be printed but not executed")
        return

    # Track failures
    failures = []

    # Run experiments
    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# SEED {seed}")
        print(f"{'#'*60}")

        # Step 1: Preprocess
        if not args.skip_preprocess:
            for split in splits:
                cmd = [
                    "python", "scripts/baseline/01_preprocess_b3db.py",
                    "--seed", str(seed),
                    "--split_type", split
                ]
                if not run_command(cmd, f"Preprocessing (seed={seed}, split={split})"):
                    failures.append(f"preprocess_seed_{seed}_split_{split}")

        # Step 2: Features
        if not args.skip_features:
            for split in splits:
                for feature in features:
                    cmd = [
                        "python", "scripts/baseline/02_compute_features.py",
                        "--seed", str(seed),
                        "--split", split,
                        "--feature", feature
                    ]
                    if not run_command(cmd, f"Features (seed={seed}, split={split}, feature={feature})"):
                        failures.append(f"features_seed_{seed}_split_{split}_feature_{feature}")

        # Step 3: Training
        for split in splits:
            for feature in features:
                cmd = [
                    "python", "scripts/baseline/03_train_baselines.py",
                    "--seed", str(seed),
                    "--split", split,
                    "--feature", feature,
                    "--models", ','.join(models)
                ]
                if not run_command(cmd, f"Training (seed={seed}, split={split}, feature={feature}, models={models})"):
                    failures.append(f"training_seed_{seed}_split_{split}_feature_{feature}")

    # Aggregate results
    print(f"\n{'='*60}")
    print("Aggregating results...")
    print('='*60)

    cmd = ["python", "scripts/analysis/aggregate_results.py"]
    run_command(cmd, "Aggregate results")

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT MATRIX COMPLETE")
    print('='*60)

    if failures:
        print(f"\n⚠️  {len(failures)} experiments failed:")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("\n✓ All experiments completed successfully!")

    print(f"\nResults saved to: artifacts/reports/")
    print(f"Master table: artifacts/reports/baseline_results_master.csv")


if __name__ == "__main__":
    main()
