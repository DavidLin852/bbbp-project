#!/usr/bin/env python
"""
Aggregate baseline experiment results.

This script collects results from multiple experiments
and creates a unified summary table.

Usage:
    python scripts/analysis/aggregate_results.py --output_dir artifacts/reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_comparison_json(json_path: Path) -> dict:
    """Load comparison.json file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_result_from_comparison(comparison: dict) -> dict:
    """
    Extract individual result from comparison.json.

    Args:
        comparison: Loaded comparison.json dict

    Returns:
        Flattened result dict
    """
    results = []

    for result in comparison.get('results', []):
        row = {
            'model_name': result.get('model_name'),
            'feature_type': result.get('feature_type'),
            'train_auc': result.get('train_auc'),
            'train_accuracy': result.get('train_accuracy'),
            'train_f1': result.get('train_f1'),
            'val_auc': result.get('val_auc'),
            'val_accuracy': result.get('val_accuracy'),
            'val_f1': result.get('val_f1'),
            'test_auc': result.get('test_auc'),
            'test_accuracy': result.get('test_accuracy'),
            'test_f1': result.get('test_f1'),
        }
        results.append(row)

    return results


def scan_results_directory(base_dir: Path) -> list[dict]:
    """
    Scan results directory for all comparison.json files.

    Args:
        base_dir: Base results directory (e.g., artifacts/models/baselines)

    Returns:
        List of all result dicts
    """
    all_results = []

    # Find all comparison.json files
    comparison_files = list(base_dir.rglob("comparison.json"))

    for comp_file in comparison_files:
        # Extract metadata from path
        # Expected: artifacts/models/baselines/seed_{seed}/{split}/{feature}/comparison.json
        parts = comp_file.parts

        try:
            # Find the index that contains 'seed_'
            seed_idx = None
            for i, part in enumerate(parts):
                if part.startswith('seed_'):
                    seed_idx = i
                    break

            if seed_idx is None:
                raise ValueError("seed_ component not found in path")

            seed = int(parts[seed_idx].replace('seed_', ''))

            split = parts[seed_idx + 1]
            feature = parts[seed_idx + 2]

        except (ValueError, IndexError):
            # Path structure doesn't match expected format
            print(f"Warning: Skipping unexpected path: {comp_file}")
            continue

        # Load comparison
        comparison = load_comparison_json(comp_file)

        # Extract results with metadata
        results = extract_result_from_comparison(comparison)

        # Add metadata to each result
        for result in results:
            result['seed'] = seed
            result['split_type'] = split
            result['feature'] = feature

        all_results.extend(results)

    return all_results


def create_master_table(results: list[dict]) -> pd.DataFrame:
    """
    Create master results table.

    Args:
        results: List of result dicts

    Returns:
        DataFrame with all results
    """
    df = pd.DataFrame(results)

    # Reorder columns
    column_order = [
        'seed', 'split_type', 'feature', 'model_name',
        'train_auc', 'train_accuracy', 'train_f1',
        'val_auc', 'val_accuracy', 'val_f1',
        'test_auc', 'test_accuracy', 'test_f1',
    ]

    # Ensure all columns exist
    for col in column_order:
        if col not in df.columns:
            df[col] = None

    df = df[column_order]

    # Sort by seed, split_type, feature, test_auc
    df = df.sort_values(['seed', 'split_type', 'feature', 'test_auc'], ascending=[True, True, True, False])

    return df


def create_summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary grouped by model.

    Args:
        df: Master results table

    Returns:
        Summary DataFrame
    """
    # Group by model_name and feature
    summary = df.groupby(['model_name', 'feature']).agg({
        'test_auc': ['mean', 'std', 'min', 'max', 'count'],
        'test_f1': ['mean', 'std'],
    }).round(4)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Reorder by mean test_auc
    summary = summary.sort_values('test_auc_mean', ascending=False)

    return summary


def create_summary_by_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary grouped by feature type.

    Args:
        df: Master results table

    Returns:
        Summary DataFrame
    """
    # Group by feature and model
    summary = df.groupby(['feature', 'model_name']).agg({
        'test_auc': ['mean', 'std'],
    }).round(4)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Reorder
    summary = summary.sort_values(['feature', 'test_auc_mean'], ascending=[True, False])

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate baseline experiment results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="artifacts/models/baselines",
        help="Base results directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/reports",
        help="Output directory for aggregated reports"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Filter by seed (default: all seeds)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Filter by split type (default: all splits)"
    )
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="Filter by feature type (default: all features)"
    )

    args = parser.parse_args()

    # Setup paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning results from {results_dir}...")

    # Scan all results
    all_results = scan_results_directory(results_dir)

    print(f"Found {len(all_results)} results")

    if len(all_results) == 0:
        print("No results found!")
        return

    # Create master table
    df = create_master_table(all_results)

    # Apply filters
    if args.seed is not None:
        df = df[df['seed'] == args.seed]
    if args.split is not None:
        df = df[df['split_type'] == args.split]
    if args.feature is not None:
        df = df[df['feature'] == args.feature]

    print(f"Filtered to {len(df)} results")

    # Save master table
    master_path = output_dir / "baseline_results_master.csv"
    df.to_csv(master_path, index=False)
    print(f"Saved master table to {master_path}")

    # Create summaries
    summary_by_model = create_summary_by_model(df)
    summary_model_path = output_dir / "baseline_summary_by_model.csv"
    summary_by_model.to_csv(summary_model_path, index=False)
    print(f"Saved model summary to {summary_model_path}")

    summary_by_feature = create_summary_by_feature(df)
    summary_feature_path = output_dir / "baseline_summary_by_feature.csv"
    summary_by_feature.to_csv(summary_feature_path, index=False)
    print(f"Saved feature summary to {summary_feature_path}")

    # Print summary
    print("\n=== Top Results by Test AUC ===")
    print(df[['seed', 'split_type', 'feature', 'model_name', 'test_auc', 'test_f1']].head(10).to_string(index=False))

    print(f"\n=== Summary by Model (Top 10) ===")
    print(summary_by_model.head(10).to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
